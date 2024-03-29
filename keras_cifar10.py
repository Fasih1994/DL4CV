from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.models import Sequential
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help="path to output loss/accuracy graph")
args = vars(ap.parse_args())

print('[INFO] loading CIFAR10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255
testX = testX.astype('float') / 255
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

## define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

# define optimizer and compile model
print('[INFO] training network...')
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              epochs=100, batch_size=32)

# evaluate network
print('[INFO] evaluating network....')
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))

# plot loss/accuracy graph
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='Training Loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='Validation loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='Training Accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='Validation Accuracy')
plt.title("Training Loss/Accuracy Graph CIFAR10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['output'])