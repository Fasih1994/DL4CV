from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to the dataset')
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))


# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader([sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) =train_test_split(data, labels,
                                                 test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")

opt = SGD(0.005)
model = ShallowNet.build(widht=32, height=32, depth=3, classes=3)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train network
print("[INFO] training network...")

H = model.fit(trainX, trainY,
          validation_data=(testX, testY),
          epochs=100, batch_size=32,
          verbose=1)
# evaluate netwrok....
print("[INFO] evaluate netwrok....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=['cat', 'dog', 'panda']))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()