# set matplotlub backend so figure can be saved in background
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from pyimagesearch.nn.conv import MiniVGGNet
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='Path to output graph for loss/accuracy  ')
args = vars(ap.parse_args())

# load and scale dataset
print('[INFO] loading dataset...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# compile model
print('[INFO] compiling model...')
opt = SGD(lr=0.01, momentum=0.9, decay=0.01/40, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])
# train
print('[INFO] training model...')
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              epochs=40, batch_size=64, verbose=1)

# evaluate
print('[INFO] evaluating model...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

#plot
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])