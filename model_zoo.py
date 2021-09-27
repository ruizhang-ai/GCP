import logging
import os

from keras.layers import *
from keras.models import model_from_json, Model
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
from keras.regularizers import l2, AlphaRegularizer
from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.extra.regularize import Regularize


class TranslationModel(Model_Wrapper):
    def piecepool(self, x):
        return x[:,:1024]

    def __init__(self, params, model_type='Translation_Model', verbose=0, structure_path=None, weights_path=None, model_name=None, vocabularies=None, store_path=None, set_optimizer=True, clear_dirs=True):
        super(TranslationModel, self).__init__(type=model_type, model_name=model_name, silence=verbose == 0, models_path=store_path, inheritance=True)
        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = model_type
        self.params = params
        self.vocabularies = vocabularies
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']
        self.return_alphas = params['COVERAGE_PENALTY'] or params['POS_UNK']
        self.setName(model_name, models_path=store_path, clear_dirs=clear_dirs)

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS'])).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
                if src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SRC_PRETRAINED_VECTORS_TRAINABLE']
            del src_word_vectors

        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = True

        # Prepare target word embedding
        '''if params['TRG_PRETRAINED_VECTORS'] is not None:
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_outputs[0]]['words2idx'].iteritems():
                if trg_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE']
            del trg_word_vectors
        else:
            self.trg_embedding_weights = None
            self.trg_embedding_weights_trainable = True'''

        # Prepare model
        if structure_path:
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, model_type):
                eval('self.' + model_type + '(params)')
            else:
                raise Exception('Translation_Model model_type "' + model_type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            self.model.summary()
        if set_optimizer:
            self.setOptimizer()

    def setParams(self, params):
        self.params = params

    def setOptimizer(self, **kwargs):
        optimizer = Adam(lr=self.params.get('LR', 0.002),
                             beta_1=self.params.get('BETA_1', 0.9),
                             beta_2=self.params.get('BETA_2', 0.999),
                             decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                             clipnorm=self.params.get('CLIP_C', 0.),
                             clipvalue=self.params.get('CLIP_V', 0.))
        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'],
                           metrics=self.params.get('KERAS_METRICS', []),
                           sample_weight_mode='temporal' if self.params['SAMPLE_WEIGHTS'] else None)

    def AttentionRNNEncoderDecoder(self, params):
        max_input_length = params['INPUT_MAX_LEN']
        max_output_length = params['OUTPUT_MAX_LEN']
        triple_length = params['TRIPLE_MAX_LEN']
        encoder_model = params['ENCODER_MODEL']
        auto_adaptive = params['AUTO_ADAPTIVE']
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], shape=(max_input_length,), dtype='int32')
        # 2. Encoder
        

        if encoder_model == "GLSTM":
            input_data = dict()
            for i in range(1, (max_input_length/triple_length)+1):
                input_data[i] = Lambda(lambda x: x[:,(i-1)*triple_length:i*triple_length],  output_shape=(triple_length,))(src_text)

            embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], 
                                params['SOURCE_TEXT_EMBEDDING_SIZE'], 
                                name='source_word_embedding', 
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']), 
                                embeddings_initializer=params['INIT_FUNCTION'], 
                                trainable=self.src_embedding_weights_trainable, 
                                weights=self.src_embedding_weights,
                                mask_zero=False)

            src_embedding = dict()
            for i in range(1, (max_input_length/triple_length)+1):
                src_embedding[i] = embedding(input_data[i])
                #src_embedding[i] = Regularize(src_embedding[i], params, name='src_embedding'+str(i))
                src_embedding[i] = Reshape((1,1500))(src_embedding[i])
	    
            first = True
            for i in range(1, (max_input_length/triple_length)):
                try:
                    if first:
                        glstm_input = concatenate([src_embedding[i], src_embedding[i+1]], axis = 1)
                        #inv_glstm_input = concatenate([src_embedding[i+1], src_embedding[i]], axis = 1)
                        first = False
                    else:
                        glstm_input = concatenate([glstm_input, src_embedding[i+1]], axis = 1)
                        #inv_glstm_input = concatenate([src_embedding[i+1], inv_glstm_input], axis = 1)
                except:
                    pass

            glstm_input = (Masking()(Reshape((-1, 1500*2))(glstm_input)))

            glstm_net = Bidirectional(GLSTM(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                         #dropout=0.5,
                                                                         #recurrent_dropout=0.5,
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True), merge_mode='concat')
            annotations = glstm_net(glstm_input)
            annotations = Masking()(annotations)
            ctx_mean = MaskedMean()(annotations)

            
            next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
            
            state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                    name='target_word_embedding',
                                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                    embeddings_initializer=params['INIT_FUNCTION'],
                                    trainable=self.src_embedding_weights_trainable, weights=self.src_embedding_weights,
                                    mask_zero=True)(next_words)
            state_below = Regularize(state_below, params, name='state_below')

        elif encoder_model == "LSTM":
            src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                      name='source_word_embedding',
                                      embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                      embeddings_initializer=params['INIT_FUNCTION'],
                                      trainable=self.src_embedding_weights_trainable, weights=self.src_embedding_weights,
                                      mask_zero=True)(src_text)
            src_embedding = Regularize(src_embedding, params, name='src_embedding')
            
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
            annotations = Masking()(annotations)
            ctx_mean = MaskedMean()(annotations)
            annotations = Regularize(annotations, params, name='annotations')
            
            next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
            
            state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                    name='target_word_embedding',
                                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                    embeddings_initializer=params['INIT_FUNCTION'],
                                    trainable=self.src_embedding_weights_trainable, weights=self.src_embedding_weights,
                                    mask_zero=True)(next_words)
            state_below = Regularize(state_below, params, name='state_below')

        else:
            input_data = dict()
            for i in range(1, (max_input_length/triple_length)+1):
                input_data[i] = Lambda(lambda x: x[:,(i-1)*triple_length:i*triple_length],  output_shape=(triple_length,))(src_text)

            embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], 
                                params['SOURCE_TEXT_EMBEDDING_SIZE'], 
                                name='source_word_embedding', 
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']), 
                                embeddings_initializer=params['INIT_FUNCTION'], 
                                trainable=self.src_embedding_weights_trainable, 
                                weights=self.src_embedding_weights,
                                mask_zero=False)

            src_embedding = dict()
            for i in range(1, (max_input_length/triple_length)+1):
                src_embedding[i] = embedding(input_data[i])
                src_embedding[i] = Regularize(src_embedding[i], params, name='src_embedding'+str(i))
            
            if encoder_model == "TFF":
                annotations = TimeDistributed(Dense(params['ENCODER_HIDDEN_SIZE'], name='encoder_' + params['ENCODER_RNN_TYPE']))

            if encoder_model == "TLSTM":
                annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=False),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')
            
            annotation = dict()
            filter_sizes = [3,4,5]
            conv_nonlinearity = Dense(params['ENCODER_HIDDEN_SIZE'])
            if encoder_model == "TCNN":
                for i in range(1, (max_input_length/triple_length)+1):
                    convs = []
                    for fsz in filter_sizes:
                        l_conv = Conv1D(params['ENCODER_HIDDEN_SIZE']/4,fsz,activation='relu', kernel_initializer=params['INIT_FUNCTION'], kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']), bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']))(src_embedding[i])
                        l_pool = Flatten()(l_conv)
                        l_pool = Lambda(self.piecepool, output_shape=lambda s: (s[0], 1024))(l_pool)
                        convs.append(l_pool)
                    first = True
                    if len(filter_sizes) > 1:
                        for j in range(0, len(filter_sizes)-1):
                            if first:
                                l_merge = concatenate([convs[j], convs[j+1]])
                                first = False
                            else:
                                l_merge = concatenate([l_merge, convs[j+1]])
                    annotation[i] = l_merge
            else:
                for i in range(1, (max_input_length/triple_length)+1):
                    annotation[i] = annotations(src_embedding[i])

            if not auto_adaptive:
                first = True
                if len(annotation) > 1:
                    for i in range(1, (max_input_length/triple_length)):
                        if first:
                            merged = concatenate([annotation[i], annotation[i+1]])
                            first = False
                        else:
                            merged = concatenate([merged, annotation[i+1]])
                if encoder_model == "TFF":
                    merged = Lambda(lambda x: K.max(x, axis=1), output_shape=lambda s: (s[0], s[2]))(merged)
            else:
                for i in range(1, (max_input_length/triple_length)+1):
                    if encoder_model == "TFF":
                        annotation[i] = Lambda(lambda x: K.max(x, axis=1), output_shape=lambda s: (s[0], s[2]))(annotation[i])
                first = True
                for i in range(1, (max_input_length/triple_length)):
                    if first:
                        merged = concatenate([annotation[i], annotation[i+1]], axis = 1)
                        first = False
                    else:
                        merged = concatenate([merged, annotation[i+1]], axis = 1)
                if encoder_model == "TLSTM":
                    merged = Reshape((max_input_length/triple_length, params['ENCODER_HIDDEN_SIZE']*2))(merged)
                elif encoder_model == "TCNN":
                    merged = Reshape((max_input_length/triple_length, 1024*len(filter_sizes)))(merged)
                else:
                    merged = Reshape((max_input_length/triple_length, params['ENCODER_HIDDEN_SIZE']))(merged)            
            
            if auto_adaptive:
                merged = Masking()(merged)
                ctx_mean = MaskedMean()(merged)
                annotations = MaskLayer()(merged)
            else:
                prev_annotations = merged
                annotations = RepeatVector(max_output_length)(merged)
                annotations = Regularize(annotations, params, name='annotations')
                ctx_mean = prev_annotations
            
            
            next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
            
            state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                    name='target_word_embedding',
                                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                    embeddings_initializer=params['INIT_FUNCTION'],
                                    trainable=self.src_embedding_weights_trainable, weights=self.src_embedding_weights,
                                    mask_zero=True)(next_words)
            state_below = Regularize(state_below, params, name='state_below')
            
        
        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init]
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1]
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1])(ctx_mean)
                initial_memory = Regularize(initial_memory, params, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'])(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        
        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=0.,
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond')


        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2))

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [proj_h, shared_Lambda_Permute(x_att), initial_state]
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)

            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=0.,
                recurrent_dropout=0.,
                conditional_dropout=0.,
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add()([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear'),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear'),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear'),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(state_below)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input')
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh')

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          ),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION']
                                               ),
                                         name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)

        self.model = Model(inputs=[src_text, next_words], outputs=softout)
        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
            self.model.add_loss(alpha_regularizer)
        ##################################################################
        #                         SAMPLING MODEL                         #
        ##################################################################
        # Now that we have the basic training model ready, let's prepare the model for applying decoding
        # The beam-search model will include all the minimum required set of layers (decoder stage) which offer the
        # possibility to generate the next state in the sequence given a pre-processed input (encoder stage)
        # First, we need a model that outputs the preprocessed input + initial h state
        # for applying the initial forward pass
        model_init_input = [src_text, next_words]
        model_init_output = [softout, annotations] + h_states_list
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            model_init_output += h_memories_list
        if self.return_alphas:
            model_init_output.append(alphas)
        self.model_init = Model(inputs=model_init_input, outputs=model_init_output)

        # Store inputs and outputs names for model_init
        self.ids_inputs_init = self.ids_inputs
        ids_states_names = ['next_state_' + str(i) for i in range(len(h_states_list))]

        # first output must be the output probs.
        self.ids_outputs_init = self.ids_outputs + ['preprocessed_input'] + ids_states_names
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            ids_memories_names = ['next_memory_' + str(i) for i in range(len(h_memories_list))]
            self.ids_outputs_init += ids_memories_names
        # Second, we need to build an additional model with the capability to have the following inputs:
        #   - preprocessed_input
        #   - prev_word
        #   - prev_state
        # and the following outputs:
        #   - softmax probabilities
        #   - next_state
        preprocessed_size = params['ENCODER_HIDDEN_SIZE'] * 2 if \
            params['BIDIRECTIONAL_ENCODER'] \
            else params['ENCODER_HIDDEN_SIZE']
        # Define inputs
        n_deep_decoder_layer_idx = 0
        if encoder_model == "LSTM":
            preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([None, preprocessed_size]))
        else:
            preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([max_output_length, preprocessed_size]))
        prev_h_states_list = [Input(name='prev_state_' + str(i),
                                    shape=tuple([params['DECODER_HIDDEN_SIZE']]))
                              for i in range(len(h_states_list))]

        input_attentional_decoder = [state_below, preprocessed_annotations,
                                     prev_h_states_list[n_deep_decoder_layer_idx]]

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            prev_h_memories_list = [Input(name='prev_memory_' + str(i),
                                          shape=tuple([params['DECODER_HIDDEN_SIZE']]))
                                    for i in range(len(h_memories_list))]

            input_attentional_decoder.append(prev_h_memories_list[n_deep_decoder_layer_idx])
        # Apply decoder
        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_states_list = [rnn_output[3]]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [rnn_output[4]]
        for reg in shared_reg_proj_h:
            proj_h = reg(proj_h)

        for (rnn_decoder_layer, proj_h_reg) in zip(shared_proj_h_list, shared_reg_proj_h_list):
            n_deep_decoder_layer_idx += 1
            input_rnn_decoder_layer = [proj_h, shared_Lambda_Permute(x_att),
                                       prev_h_states_list[n_deep_decoder_layer_idx]]
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_rnn_decoder_layer.append(prev_h_memories_list[n_deep_decoder_layer_idx])

            current_rnn_output = rnn_decoder_layer(input_rnn_decoder_layer)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])  # h_state
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])  # h_memory
            for reg in proj_h_reg:
                current_proj_h = reg(current_proj_h)
            proj_h = Add()([proj_h, current_proj_h])
        out_layer_mlp = shared_FC_mlp(proj_h)
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        out_layer_emb = shared_FC_emb(state_below)

        for (reg_out_layer_mlp, reg_out_layer_ctx, reg_out_layer_emb) in zip(shared_reg_out_layer_mlp,
                                                                             shared_reg_out_layer_ctx,
                                                                             shared_reg_out_layer_emb):
            out_layer_mlp = reg_out_layer_mlp(out_layer_mlp)
            out_layer_ctx = reg_out_layer_ctx(out_layer_ctx)
            out_layer_emb = reg_out_layer_emb(out_layer_emb)

        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        out_layer = shared_activation_tanh(additional_output)

        for (deep_out_layer, reg_list) in zip(shared_deep_list, shared_reg_deep_list):
            out_layer = deep_out_layer(out_layer)
            for reg in reg_list:
                out_layer = reg(out_layer)

        # Softmax
        softout = shared_FC_soft(out_layer)
        model_next_inputs = [next_words, preprocessed_annotations] + prev_h_states_list
        model_next_outputs = [softout, preprocessed_annotations] + h_states_list
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            model_next_inputs += prev_h_memories_list
            model_next_outputs += h_memories_list

        if self.return_alphas:
            model_next_outputs.append(alphas)

        self.model_next = Model(inputs=model_next_inputs,
                                outputs=model_next_outputs)
        # Store inputs and outputs names for model_next
        # first input must be previous word
        self.ids_inputs_next = [self.ids_inputs[1]] + ['preprocessed_input']
        # first output must be the output probs.
        self.ids_outputs_next = self.ids_outputs + ['preprocessed_input']
        # Input -> Output matchings from model_init to model_next and from model_next to model_next
        self.matchings_init_to_next = {'preprocessed_input': 'preprocessed_input'}
        self.matchings_next_to_next = {'preprocessed_input': 'preprocessed_input'}
        # append all next states and matchings

        for n_state in range(len(prev_h_states_list)):
            self.ids_inputs_next.append('prev_state_' + str(n_state))
            self.ids_outputs_next.append('next_state_' + str(n_state))
            self.matchings_init_to_next['next_state_' + str(n_state)] = 'prev_state_' + str(n_state)
            self.matchings_next_to_next['next_state_' + str(n_state)] = 'prev_state_' + str(n_state)

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            for n_memory in range(len(prev_h_memories_list)):
                self.ids_inputs_next.append('prev_memory_' + str(n_memory))
                self.ids_outputs_next.append('next_memory_' + str(n_memory))
                self.matchings_init_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)
                self.matchings_next_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)

    # Backwards compatibility.
    GroundHogModel = AttentionRNNEncoderDecoder