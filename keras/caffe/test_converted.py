from keras.models import Sequential, model_from_json
from keras.optimizers import SGD

from scipy import misc
import numpy as np
import copy

if __name__ == "__main__":

    out_layer_names = ["loss1/loss", "loss2/loss", "loss3/loss3"]

    print "Preparing test image."
    # Read image
    im = misc.imread('models/cat.jpg')

    # Resize
    im = misc.imresize(im, (224, 224)).astype(np.float32)
    
    # Change RGB to BGR
    aux = copy.copy(im)
    im[:,:,0] = aux[:,:,2]
    im[:,:,2] = aux[:,:,0]

    # Remove train image mean
    im[:,:,0] -= 104.006
    im[:,:,1] -= 116.669
    im[:,:,2] -= 122.679

    # Transpose image dimensions (Keras' uses the channels as the 1st dimension)
    im = np.transpose(im, (2, 0, 1))

    # Insert a new dimension for the batch_size
    im = np.expand_dims(im, axis=0)


    # Load the converted model
    print "Loading model."
    # Load model structure
    model = model_from_json(open('models/Keras_model_structure.json').read())
    # Load model weights
    model.load_weights('models/Keras_model_weights.h5') 

    # Compile converted model
    print "Compiling model."
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    loss = dict()
    for out in out_layer_names:
        loss[out] = 'categorical_crossentropy'
        last_out = out
    model.compile(optimizer=sgd, loss=loss)
    
    # Predict image output
    print "Applying prediction."
    in_data = dict()
    for input in ['data']:
        in_data[input] = im
    out = model.predict(in_data)

    # Load ImageNet classes file
    classes = []
    with open('models/classes.txt', 'r') as list_:
        for line in list_:
            classes.append(line.rstrip('\n'))

    for i, o in enumerate(out_layer_names):
        print 'Prediction on output layer "'+o+'": '+str(classes[np.argmax(out[i])])
