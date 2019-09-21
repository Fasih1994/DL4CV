from pyimagesearch.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load mnist dataset
print('[INFO] loadning mnist dataset...')
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data -data.min()) / (data.max() - data.min())
print('[INFO] samples={}, dim={}'.format(data.shape[0],data.shape[1]))

# make train test splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

#encode targets
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))

nn.fit(trainX, trainY, epochs=1000)

# evaluate model
print('[INFO] evaluating network...')

predictions = nn.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

# plot loss graph
plt.style.use('ggplot')
plt.figure()
plt.plot(range(0, 1100, 100), nn.get_losses())
plt.title("Losses w.r.t epochs")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.show()