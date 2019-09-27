from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from pyimagesearch.nn.conv import LeNet
import argparse
import numpy as np
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-o', '--output', required=True,
                help='path for output model')
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in list(paths.list_images(args['dataset'])):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not smiling'
    labels.append(label)

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

lb = LabelBinarizer().fit(labels)
labels = np_utils.to_categorical(lb.transform(labels), num_classes=2)

# takeing care for class imbalance
classTotals = labels.sum(axis=0)  # [9475, 3690]
classWeights = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data, labels, stratify=labels,
                                                  test_size=0.20, random_state=42)

print('[INFO] compiling model...')
checkpoints = ModelCheckpoint(args['output'], monitor='val_loss', save_best_only=True,
                              mode='min', verbose=1)
model = LeNet.build(width=28, height=28, depth=1, classes=2)

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

print('[INFO] training model...')

H = model.fit(trainX, trainY, validation_data=(testX, testY),
              class_weight=classWeights, batch_size=64, verbose=1,
              epochs=15)
print('[INFO] Serializing model...')
model.save(args['output'])
print('[INFO] evaluating model...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()