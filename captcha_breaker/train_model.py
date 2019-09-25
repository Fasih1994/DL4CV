from pyimagesearch.nn.conv import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagesearch.utils.captchahelper import preprocess
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to dataset directory')
ap.add_argument('-m', '--model', required=True,
                help='path to output model')
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in list(paths.list_images(args['dataset'])):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, width=28, height=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)


data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling the model...")
model = LeNet.build(28, 28, 1, 9)
opt = SGD(lr=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['acc'])


print('[INFO] training model...')
H = model.fit(trainX, trainY,
              validation_data=(testX, testY), epochs=15,
              batch_size=32, verbose=1)

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))
print('[INFO] serializing model...')
model.save(args['model'])



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