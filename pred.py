from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from keras.datasets import cifar10
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True,
                help='path to weights directory')
args = vars(ap.parse_args())
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
((_, _), (testX, testY)) = cifar10.load_data()
testY = LabelBinarizer().fit_transform(testY)
testX = testX.astype('float') / 255.0
model = load_model(args['weights'])
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))