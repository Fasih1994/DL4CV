from pyimagesearch.nn import NeuralNetwork
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define a neural network
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=6500)

# print results
preds = []
for (x,target) in zip(X,y):
    #predict result then apply threshold
    pred = nn.predict(x)[0][0]
    preds.append([pred])
    #apply step function
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, prediction={:.7f}, ground_truth={}, step={}".format(x, pred, target[0], step))
print("[INFO] network architecture is: {}".format(nn.__repr__()))