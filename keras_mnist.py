import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.models import Sequential

#construct parser
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the full MNIST dataset
print("[INFO] loading mnist (full) dataset...")
dataset = datasets.fetch_openml("mnist_784")

# scale raw pixel intensities in range [0, 1.0]
data = dataset.data.astype('float') /255

#construct train test splits
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# DO ONE-HOT ENCODE | construct label from integer to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define 784-256-128-10 architecture using keras
model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# Train the network
print("[INFO] training network...")
sgd = SGD(lr=0.01)
# compile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(trainX, trainY,
                    validation_data=(testX, testY),
                    epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot loss/accuracy graph
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), history.history['loss'], label='Training Loss')
plt.plot(np.arange(0, 100), history.history['val_loss'], label='Validation Loss')
plt.plot(np.arange(0, 100), history.history['acc'], label='Training Accuracy')
plt.plot(np.arange(0, 100), history.history['val_acc'], label='Validation Accuracy')
plt.title("Loss/Accuracy Graph")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['output'])