from pyimagesearch.nn.conv import LeNet
from keras.utils import plot_model

# initialize LeNet model then write
# network architecture graph to disk
model = LeNet.build(width=28, height=28, depth=1, classes=10)
plot_model(model, to_file='architecture/LeNet.png', show_shapes=True)