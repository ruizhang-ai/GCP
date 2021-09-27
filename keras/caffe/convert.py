import numpy as np
from google.protobuf import text_format

from . import caffe_pb2 as caffe
from .caffe_utils import *
from .extra_layers import *
from ..layers import *
from ..models import Model


def caffe_to_keras(prototext, caffemodel, phase='train', debug=False):
    '''
        Converts a Caffe Graph into a Keras Graph
        prototext: model description file in caffe
        caffemodel: stored weights file
        phase: train or test

        Usage:
            model = caffe_to_keras('VGG16.prototxt', 'VGG16_700iter.caffemodel')
    '''
    config = caffe.NetParameter()
    prototext = preprocessPrototxt(prototext, debug)
    text_format.Merge(prototext, config)

    if len(config.layers) != 0:
        raise Exception("Prototxt files V1 are not supported.")
        layers = config.layers[:]  # prototext V1
    elif len(config.layer) != 0:
        layers = config.layer[:]  # prototext V2
    else:
        raise Exception('could not load any layers from prototext')

    input_dim = []
    if len(config.input_dim[:]) == 0:
        input_dim.append(int(layers[0].input_param.shape[0].dim[0]))
        input_dim.append(int(layers[0].input_param.shape[0].dim[1]))
        input_dim.append(int(layers[0].input_param.shape[0].dim[2]))
        input_dim.append(int(layers[0].input_param.shape[0].dim[3]))
    else:
        input_dim = tuple(config.input_dim[:])
    print layers
    print("CREATING MODEL")
    model = create_model(layers,
                         0 if phase == 'train' else 1,
                         tuple(input_dim[1:]), debug)
    params = caffe.NetParameter()
    params.MergeFromString(open(caffemodel, 'rb').read())

    if len(params.layers) != 0:
        param_layers = params.layers[:]  # V1
        v = 'V1'
    elif len(params.layer) != 0:
        param_layers = params.layer[:]  # V2
        v = 'V2'
    else:
        raise Exception('could not load any layers from caffemodel')

    print "Printing the converted model:"
    model.summary()

    print('')
    print("LOADING WEIGHTS")
    weights = convert_weights(param_layers, v, debug)

    load_weights(model, weights)

    return model


def preprocessPrototxt(prototxt, debug=False):
    p = open(prototxt).read().split('\n')

    for i, line in enumerate(p):
        l = line.strip().replace(" ", "").split('#')[0]
        # Change "layers {" to "layer {"
        # if len(l) > 6 and l[:7] == 'layers{':
        #    p[i] = 'layer {'
        # Write all layer types as strings
        if len(l) > 6 and l[:5] == 'type:' and l[5] != "\'" and l[5] != '\"':
            type_ = l[5:]
            p[i] = '  type: "' + type_ + '"'
            # blobs_lr
            # elif len(l) > 9 and l[:9] == 'blobs_lr:':
            #    print("The prototxt parameter 'blobs_lr' found in line "+str(i+1)+" is outdated and will be removed. Consider using param { lr_mult: X } instead.")
            #    p[i] = ''
            #
            # elif len(l) > 13 and l[:13] == 'weight_decay:':
            #    print("The prototxt parameter 'weight_decay' found in line "+str(i+1)+" is outdated and will be removed. Consider using param { decay_mult: X } instead.")
            #    p[i] = ''

    p = '\n'.join(p)
    if debug:
        print 'Writing preprocessed prototxt to debug.prototxt'
        f = open('debug.prototxt', 'w')
        f.write(p)
        f.close()
    return p


def create_model(layers, phase, input_dim, debug=False):
    '''
        layers:
            a list of all the layers in the model
        phase:
            parameter to specify which network to extract: training or test
        input_dim:
            `input dimensions of the configuration (if in model is in deploy mode)
    '''
    # DEPLOY MODE: first layer is computation (conv, dense, etc..)
    # NON DEPLOY MODE: first layer is data input
    if input_dim == ():
        in_deploy_mode = False
    else:
        in_deploy_mode = True

    # obtain the nodes that make up the graph
    # returned in linked list (not matrix) representation (dictionary here)
    network = parse_network(layers, phase)

    if len(network) == 0:
        raise Exception('failed to construct network from the prototext')

    # inputs of the network - 'in-order' is zero
    inputs = get_inputs(network)
    # outputs of the network - 'out-order' is zero
    network_outputs = get_outputs(network)

    # path from input to loss layers (label) removed
    network = remove_label_paths(layers, network, inputs, network_outputs)

    # while network contains what nodes follow a particular node.
    # we need to know what feeds a given node, hence reverse it.
    inputs_to = reverse(network)

    # create all net nodes without link
    net_node = [None] * (max(network) + 1)
    for n_layer, layer_nb in enumerate(network):
        layer = layers[layer_nb]
        name = layer.name
        type_of_layer = layer_type(layer)

        # case of inputs
        if layer_nb in inputs:
            if in_deploy_mode:
                dim = input_dim
            else:
                # raise Exception("You must define the 'input_dim' of your network at the start of your .prototxt file.")
                dim = get_data_dim(layers[0])
            net_node[layer_nb] = Input(shape=dim, name=name)

        # other cases
        else:
            input_layers = [None] * (len(inputs_to[layer_nb]))
            for l in range(0, len(inputs_to[layer_nb])):
                input_layers[l] = net_node[inputs_to[layer_nb][l]]

            # input_layers = net_node[inputs_to[layer_nb]]
            input_layer_names = []
            for input_layer in inputs_to[layer_nb]:
                input_layer_names.append(layers[input_layer].name)

            if debug:
                print "Layer", str(n_layer) + ":", name
                print '\t input shape: ' + str(input_layers[0]._keras_shape)

            if type(input_layers) is list and len(input_layers) == 1:
                input_layers = input_layers[0]
            if type_of_layer == 'concat':
                axis = layer.concat_param.axis
                net_node[layer_nb] = merge(input_layers, mode='concat', concat_axis=1, name=name)

            elif type_of_layer == 'convolution':
                has_bias = layer.convolution_param.bias_term
                nb_filter = layer.convolution_param.num_output
                nb_col = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_h])[0]
                nb_row = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_w])[0]
                stride_h = (layer.convolution_param.stride or [layer.convolution_param.stride_h])[0] or 1
                stride_w = (layer.convolution_param.stride or [layer.convolution_param.stride_w])[0] or 1
                pad_h = (layer.convolution_param.pad or [layer.convolution_param.pad_h])[0]
                pad_w = (layer.convolution_param.pad or [layer.convolution_param.pad_w])[0]

                if debug:
                    print("\t kernel: " + str(nb_filter) + 'x' + str(nb_col) + 'x' + str(nb_row))
                    print("\t stride: " + str(stride_h))
                    print("\t pad_h: " + str(pad_h))
                    print("\t pad_w:" + str(pad_w))
                if pad_h + pad_w > 0:
                    input_layers = ZeroPadding2D(padding=(int(pad_h), int(pad_w)), name=name + '_zeropadding')(
                        input_layers)
                if (layer.convolution_param.dilation or [layer.convolution_param.dilation])[0]:
                    dilation = layer.convolution_param.dilation[0]
                    net_node[layer_nb] = AtrousConvolution2D(nb_filter, (int(nb_row), int(nb_col)), use_bias=True,
                                                             strides=(stride_h, stride_w),
                                                             atrous_rate=(dilation, dilation), name=name,
                                                             padding='valid')(input_layers)
                else:
                    net_node[layer_nb] = Convolution2D(nb_filter, (int(nb_row), int(nb_col)), use_bias=has_bias,
                                                       strides=(stride_h, stride_w), name=name, padding='valid')(
                        input_layers)


                net_node[layer_nb] = Convolution2D(nb_filter, (int(nb_row), int(nb_col)), use_bias=has_bias,
                                                   strides=(stride_h, stride_w), name=name, padding='valid')(
                    input_layers)

            elif type_of_layer == 'deconvolution':
                has_bias = layer.convolution_param.bias_term
                nb_filter = int(layer.convolution_param.num_output)
                nb_col = int(layer.convolution_param.kernel_size[0])
                nb_row = int(layer.convolution_param.kernel_size[0])
                stride_h = int(layer.convolution_param.stride[0])
                stride_w = int(layer.convolution_param.stride[0])
                pad_h = 0
                pad_w = 0
                try:
                    pad_h = int(layer.convolution_param.pad[0])
                    pad_w = int(layer.convolution_param.pad[0])
                except Exception as e:
                    pass

                if debug:
                    print("\t Deconv kernel: " + str(nb_filter) + 'x' + str(nb_col) + 'x' + str(nb_row))
                    print("\t stride:" + str(stride_h))
                    print("\t pad_h: " + str(pad_h))
                    print("\t pad_w:" + str(pad_w))
                # shape inference
                semi_model = Model(inputs=net_node[0], outputs=input_layers)
                ip_shape = semi_model.layers[-1].output_shape
                del semi_model

                ##### FORMULA FOR O/P SHAPE OF DECONV ########
                # o = s (i - 1) + a + k - 2p
                # where:
                # i - input size (rows or cols),
                # k - kernel size (nb_filter),
                # s - stride (subsample for rows or cols respectively),
                # p - padding size
                # a - (not used)
                ##############################################

                i_h, i_w = ip_shape[2], ip_shape[3]
                output_shape = [None,
                                nb_filter,
                                stride_h * (i_h - 1) + nb_row - 2 * pad_h,
                                stride_w * (i_w - 1) + nb_col - 2 * pad_w
                                ]
                if pad_h + pad_w > 0:
                    input_layers = ZeroPadding2D(padding=(int(pad_h), int(pad_w)), name=name + '_zeropadding')(
                        input_layers)
                net_node[layer_nb] = Deconvolution2D(nb_filter, (int(nb_row), int(nb_col)),
                                                     strides=(stride_h, stride_w),
                                                     #output_shape=output_shape,
                                                     name=name, use_bias=has_bias)(input_layers)

            elif type_of_layer == "crop":
                assert (len(input_layers) == 2), "Caffe crop layer must have only 2 Bottom blobs"

                # shape inference - Input Layer [1]
                semi_model = Model(inputs=net_node[0], outputs=input_layers[0])
                shape1 = semi_model.layers[-1].output_shape
                del semi_model

                # shape inference - Input Layer [2]
                semi_model = Model(inputs=net_node[0], outputs=input_layers[1])
                shape2 = semi_model.layers[-1].output_shape
                del semi_model

                # offset parameter
                offset = int(layer.crop_param.offset[0])

                crop_param = (
                    (offset, int(shape1[2]) - (offset + int(shape2[2]))),
                    (offset, int(shape1[3]) - (offset + int(shape2[3])))
                )
                net_node[layer_nb] = Cropping2D(cropping=crop_param, name=name)(input_layers[1])

            elif type_of_layer == 'dropout':
                prob = layer.dropout_param.dropout_ratio
                net_node[layer_nb] = Dropout(prob, name=name)(input_layers)

            elif type_of_layer == 'flatten':
                net_node[layer_nb] = Flatten(name=name)(input_layers)

            elif type_of_layer == 'innerproduct':
                output_dim = layer.inner_product_param.num_output

                if len(input_layers[0]._keras_shape[1:]) > 1:
                    input_layers = Flatten(name=name + '_flatten')(input_layers)
                    input_layer_name = name + '_flatten'

                net_node[layer_nb] = Dense(output_dim, name=name)(input_layers)

            elif type_of_layer == 'lrn':
                alpha = layer.lrn_param.alpha
                k = layer.lrn_param.k
                beta = layer.lrn_param.beta
                n = layer.lrn_param.local_size
                net_node[layer_nb] = LRN2D(alpha=alpha, k=k, beta=beta, n=n, name=name)(input_layers)

            elif type_of_layer == 'pooling':
                kernel_h = layer.pooling_param.kernel_size or layer.pooling_param.kernel_h
                kernel_w = layer.pooling_param.kernel_size or layer.pooling_param.kernel_w

                # caffe defaults to 1, hence both of the params can be zero. 'or 1'
                stride_h = layer.pooling_param.stride or layer.pooling_param.stride_h or 1
                stride_w = layer.pooling_param.stride or layer.pooling_param.stride_w or 1

                pad_h = layer.pooling_param.pad or layer.pooling_param.pad_h
                pad_w = layer.pooling_param.pad or layer.pooling_param.pad_w


                if debug:
                    print("\t kernel: " + str(kernel_h) + 'x' + str(kernel_w))
                    print("\t stride: " + str(stride_h))
                    print("\t pad_h: " + str(pad_h))
                    print("\t pad_w:" + str(pad_w))

                if pad_h + pad_w > 0:
                    input_layers = ZeroPadding2D(padding=(pad_h, pad_w), name=name + '_zeropadding')(input_layers)
                    input_layer_name = name + '_zeropadding'
                if layer.pooling_param.pool == 0:  # MAX pooling
                    # border_mode = 'same'
                    border_mode = 'valid'
                    net_node[layer_nb] = MaxPooling2D(pool_size=(kernel_h, kernel_w), strides=(stride_h, stride_w),
                                                      padding=border_mode, name=name)(input_layers)
                    if debug:
                        print("\t MAX pooling")
                elif layer.pooling_param.pool == 1:  # AVE pooling
                    net_node[layer_nb] = AveragePooling2D(pool_size=(kernel_h, kernel_w), strides=(stride_h, stride_w),
                                                          name=name)(input_layers)
                    if debug:
                        print("\t AVE pooling")
                else:  # STOCHASTIC?
                    raise NotImplementedError("Only MAX and AVE pooling are implemented in keras!")

            elif type_of_layer == 'relu':
                net_node[layer_nb] = Activation('relu', name=name)(input_layers)

            elif type_of_layer == 'sigmoid':
                net_node[layer_nb] = Activation('sigmoid', name=name)(input_layers)

            elif type_of_layer == 'softmax' or type_of_layer == 'softmaxwithloss':
                # check output shape
                semi_model = Model(inputs=net_nod[0], outputs=input_layers)
                op_shape = semi_model.layers[-1].output_shape
                del semi_model

                if len(op_shape) == 4:  # for img segmentation - i/p to softmax is (None, num_classes, height, width)
                    interm_layer = Reshape((op_shape[1], op_shape[2] * op_shape[3]))(input_layers)
                    input_layers = Permute((2, 1))(interm_layer)  # reshaped to (None, height*width, num_classes)

                net_node[layer_nb] = Activation('softmax', name=name)(input_layers)

            elif type_of_layer == 'split':
                net_node[layer_nb] = Activation('tanh', name=name)(input_layers)

            elif type_of_layer == 'tanh':
                net_node[layer_nb] = Activation('tanh', name=name)(input_layers)

            elif type_of_layer == 'batchnorm':
                axis = layer.scale_param.axis
                epsilon = layer.batch_norm_param.eps
                moving_average = layer.batch_norm_param.moving_average_fraction  # unused

                if debug:
                    print '\t -- BatchNormalization'
                    print '\t axis: ' + str(axis)

                net_node[layer_nb] = BatchNormalization(epsilon=epsilon, axis=axis, name=name)(input_layers)

            elif type_of_layer == 'scale':
                axis = layer.scale_param.axis

                if debug:
                    print '\t -- Scale'
                    print '\t axis: ' + str(axis)

                net_node[layer_nb] = Scale(axis=axis, name=name)(input_layers)

            elif type_of_layer == 'eltwise':
                axis = layer.scale_param.axis
                op = layer.eltwise_param.operation  # PROD=0, SUM=1, MAX=2
                if op == 0:
                    net_node[layer_nb] = Multiply(name=name)(input_layers)
                elif op == 1:
                    net_node[layer_nb] = Add(name=name)(input_layers)
                elif op == 2:
                    net_node[layer_nb] = Maximum(name=name)(input_layers)
                else:
                    raise NotImplementedError('Operation with id = "' + str(op) +
                                              '" of layer with type "' + type_of_layer + '" is not implemented.')

            else:
                raise RuntimeError('layer type', type_of_layer, 'used in this model is not currently supported')

    input_l = [None] * (len(inputs))
    output_l = [None] * (len(network_outputs))

    for i in range(0, len(inputs)):
        input_l[i] = net_node[inputs[i]]
    for i in range(0, len(network_outputs)):
        output_l[i] = net_node[network_outputs[i]]

    model = Model(inputs=input_l, outputs=output_l)
    return model


def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


def convert_weights(param_layers, v='V1', debug=False):
    weights = {}

    for layer in param_layers:
        typ = layer_type(layer)
        if typ == 'innerproduct':
            blobs = layer.blobs

            if v == 'V1':
                nb_filter = blobs[0].num
                stack_size = blobs[0].channels
                nb_col = blobs[0].height
                nb_row = blobs[0].width
            elif v == 'V2':
                if len(blobs[0].shape.dim) == 4:
                    nb_filter = int(blobs[0].shape.dim[0])
                    stack_size = int(blobs[0].shape.dim[1])
                    nb_col = int(blobs[0].shape.dim[2])
                    nb_row = int(blobs[0].shape.dim[3])
                else:
                    nb_filter = 1
                    stack_size = 1
                    nb_col = int(blobs[0].shape.dim[0])
                    nb_row = int(blobs[0].shape.dim[1])
            else:
                raise RuntimeError('incorrect caffemodel version "' + v + '"')

            weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :]
            weights_p = weights_p.T  # need to swapaxes here, hence transpose. See comment in conv
            weights_b = np.array(blobs[1].data)
            layer_weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]

            weights[layer.name] = layer_weights

        elif typ == 'batchnorm':
            blobs = layer.blobs
            if v == 'V2':
                nb_kernels = int(blobs[0].shape.dim[0])
            else:
                raise NotImplementedError(
                    'Conversion on layer type "' + typ + '"not implemented forcaffemodel version "' + v + '"')

            weights_mean = np.array(blobs[0].data)
            weights_std_dev = np.array(blobs[1].data)

            weights[layer.name] = [np.ones(nb_kernels), np.zeros(nb_kernels), weights_mean.astype(dtype=np.float32),
                                   weights_std_dev.astype(dtype=np.float32)]

        elif typ == 'scale':
            blobs = layer.blobs
            if v == 'V2':
                nb_gamma = int(blobs[0].shape.dim[0])
                nb_beta = int(blobs[1].shape.dim[0])
                assert nb_gamma == nb_beta
            else:
                raise NotImplementedError(
                    'Conversion on layer type "' + typ + '"not implemented forcaffemodel version "' + v + '"')

            weights_gamma = np.array(blobs[0].data)
            weights_beta = np.array(blobs[1].data)

            weights[layer.name] = [weights_gamma.astype(dtype=np.float32), weights_beta.astype(dtype=np.float32)]

        elif typ == 'convolution' or typ == 'deconvolution':
            blobs = layer.blobs

            if v == 'V1':
                nb_filter = blobs[0].num
                temp_stack_size = blobs[0].channels
                nb_col = blobs[0].height
                nb_row = blobs[0].width
            elif v == 'V2':
                nb_filter = int(blobs[0].shape.dim[0])
                temp_stack_size = int(blobs[0].shape.dim[1])
                nb_col = int(blobs[0].shape.dim[2])
                nb_row = int(blobs[0].shape.dim[3])
            else:
                raise RuntimeError('incorrect caffemodel version "' + v + '"')

            # NOTE: on model parallel networks
            # if group is > 1, that means the conv filters are split up
            # into a number of 'groups' and each group lies on a seperate GPU.
            # Each group only acts on the select group of outputs from pervious layer
            # that was in the same GPU (not the entire stack)
            # Here, we add zeros to simulate the same effect
            # This was famously used in AlexNet and few other models from 2012-14

            group = layer.convolution_param.group
            stack_size = temp_stack_size * group

            weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row))

            if layer.convolution_param.bias_term:
                weights_b = np.array(blobs[1].data)
            else:
                weights_b = np.zeros((nb_filter,))

            group_data_size = len(blobs[0].data) // group
            stacks_size_per_group = stack_size // group
            nb_filter_per_group = nb_filter // group

            if debug:
                print (layer.name)
                print ("nb_filter")
                print (nb_filter)
                print ("(channels x height x width)")
                print ("(" + str(temp_stack_size) + " x " + str(nb_col) + " x " + str(nb_row) + ")")
                print ("groups")
                print (group)

            for i in range(group):
                group_weights = weights_p[i * nb_filter_per_group: (i + 1) * nb_filter_per_group,
                                i * stacks_size_per_group: (i + 1) * stacks_size_per_group, :, :]
                group_weights[:] = np.array(blobs[0].data[i * group_data_size:
                (i + 1) * group_data_size]).reshape(group_weights.shape)

            # caffe, unlike theano, does correlation not convolution. We need to flip the weights 180 deg
            weights_p = rot90(weights_p)

            if weights_b is not None:
                layer_weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]
            else:
                layer_weights = [weights_p.astype(dtype=np.float32)]

            weights[layer.name] = layer_weights

    return weights


def load_weights(model, weights):
    for layer in model.layers:
        if weights.has_key(layer.name):
            model.get_layer(layer.name).set_weights(weights[layer.name])
            print "Copied wts for layer:", layer.name
