from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True,
                help='path to weights directory')
args = vars(ap.parse_args())

print('[INFO] loading cifar10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

print("[INFO] compiling model...")

opt = SGD(lr=0.01, momentum=0.9, decay=0.01/40, nesterov=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# construct the callback to save only the *best* model to disk
# based on the validation loss
file_name = os.path.sep.join([args['weights'], "weights-{epoch:03d}-{val_loss:.5f}.hdf5"])
checkpoints = [ModelCheckpoint(filepath=file_name, monitor='val_loss', mode='min',
                               save_best_only=True, verbose=1)]
callbacks = checkpoints

print('[INFO] training model...')
model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=40, batch_size=64, callbacks=callbacks, verbose=1)
