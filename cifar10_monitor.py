import matplotlib
matplotlib.use('Agg')

from pyimagesearch.callbacks import TrainingMonitor
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='Output path for loss/accuracy plots')
args = vars(ap.parse_args())

# show information of the process ID
print("[INFO process ID: {}".format(os.getpid()))

# load training data
print('[INFO] loading Cifar10...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print('[INFO] compiling model...')

opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

figPath = os.path.sep.join([args['output'], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args['output'], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath)]

print("[INFO] training network...")
model.fit(trainX, trainY,
          validation_data=(testX, testY),
          epochs=100, verbose=1,
          batch_size=64, callbacks=callbacks)
