
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
#from scipy.misc import imsave, imresize
from PIL import Image
import skimage.transform
#import imageio
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "cat.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "starry_night.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3

tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()

#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

'''
individual layer loss is 1 / (4 * (number of filters * filter width * filter height)^2) * sum((gen style representation - original style representation in layer)^2)
'''
def styleLoss(style, gen):
    #TODO fix imgh, imgw, and numfilters, do for each layer

    #individual layer
    return K.sum(K.square(gramMatrix(gen) - gramMatrix(style)))# / (4. * numFilters**2 * ((imgH * imgW)**2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    return None   #TODO: implement.





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #img = imresize(img, (ih, iw, 3))
        #img = np.array(Image.fromarray(img).resize(size=(ih, iw, 3)))
        img = np.array(skimage.transform.resize(img, (ih, iw, 3), order=3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def calcLoss(x, genTensor, inputTensor, lossTensor, model):
    sess = tf.compat.v1.Session()
    x = np.reshape(x, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    K.update(genTensor, x)
    model.predict(inputTensor, steps=1)
    output = []
    with tf.compat.v1.Session() as sess:
        output = sess.run(genTensor)
    return output

def calcLossGradient(x, lossTensor, inputTensor):
    return None


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    #genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    genTensor = K.variable(np.zeros(shape=(1, CONTENT_IMG_H, CONTENT_IMG_W, 3)))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet");
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]

    ALPHA = 0.1 #CONTENT_WEIGHT
    BETA = 0.9 #STYLE_WEIGHT

    loss += ALPHA * contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        thisstyleLayer = outputDict[layerName]
        thisstyleOutput = thisstyleLayer[1, :, :, :]
        thisgenOutput = thisstyleLayer[2, :, :, :]
        loss += BETA * styleLoss(thisstyleOutput, thisgenOutput)
    #loss += None   #TODO: implement.
    #loss_fn = K.function([], [loss])
    # TODO: Setup gradients or use K.gradients().
    with tf.GradientTape() as tape:
        tape.watch(loss)
    print("   Beginning transfer.")
    #grads = tf.GradientTape().gradient()
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        fmin_l_bfgs_b(calcLoss, np.empty(genTensor.shape), args=(genTensor, inputTensor, loss, model), fprime=None, maxiter=15000)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x)
        saveFile = None   #TODO: Implement.
        #imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()

