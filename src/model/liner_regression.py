"""
Implementing a linear regression using Keras
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import numpy as np


# generate data
w_true = 2
b_true = 1

X = np.arange(0, 100)
Y_true = w_true * X + b_true
Y_noise = Y_true + np.random.normal(10, 10, 100)
# scale = abs(Y_noise).max()
# Y_noise = Y_noise / scale

indices = np.random.permutation(X.shape[0])
train_indices, test_indices = indices[:80], indices[80:]
X_train, X_test = X[train_indices], X[test_indices]
Y_train = Y_noise[train_indices]
Y_test = Y_true[test_indices]


model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
# if learning rate is bigger (e.g, 0.01), the weight will be nan finally
sgd = SGD(lr=0.00001, momentum=0, decay=0.01)
model.compile(loss='mse', optimizer=sgd)
model.fit(X_train, Y_train, batch_size=10, nb_epoch=20)
Y_predict = model.predict(X_test)

print model.summary()
print model.get_weights()

plt.scatter(X, Y_noise)
plt.plot(X_test, Y_predict, 'r-')
plt.show()





