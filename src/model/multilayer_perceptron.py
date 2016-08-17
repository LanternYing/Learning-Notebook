'''
test the impact of optimizer and learning rates multilayer perceptron
referece: http://cs231n.github.io/neural-networks-case-study/
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import probas_to_classes


def data():
    N = 300 # number of points per class
    D = 2   # dimension
    K = 3   # number of class
    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')
    for i in xrange(K):
        r = np.linspace(0.0, 1, N)    # radius
        t = np.linspace(i*4, (i+1)*4, N) + np.random.randn(N) * 0.2 #theta
        ix = np.arange(i*N, (i+1)*N)
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = i
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    return X, y


def train(X, y):
    y = to_categorical(y, 3)
    model = Sequential()
    model.add(Dense(100, input_dim=2, activation='relu'))
    model.add(Dense(output_dim=3, activation='softmax'))

    # test the performance of optimizer
    # optimizers = {}
    # optimizers['sgd_simple'] = SGD(lr=0.1, momentum=0, decay=0.001, nesterov=False)
    # optimizers['sgd_momentum'] = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=False)
    # optimizers['sgd_nesterov'] = SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)
    # optimizers['adam'] = Adam(lr=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-08)

    # for i, optimizer in enumerate(optimizers):
    #     model.compile(loss='categorical_crossentropy',
    #                   optimizer=optimizers[optimizer],
    #                   metrics=['accuracy'])
    #     print model.summary()
    #     history = model.fit(X, y, nb_epoch=1000, batch_size=16, verbose=0)
    #     plt.figure(num=i, figsize=(15, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(history.history['loss'])
    #     predict_y = model.predict(X, batch_size=16)
    #     predict_y = probas_to_classes(predict_y)
    #     plt.subplot(1, 2, 2)
    #     plt.scatter(X[:, 0], X[:, 1], c=predict_y, s=40, cmap=plt.cm.Spectral)
    #     plt.suptitle(optimizer)
    # plt.show()


    # test the impact of learning rate
    learning_rates = 10 ** np.arange(-1, 2, dtype='float32')
    print learning_rates
    for lr in learning_rates:
        optimize = SGD(lr=lr, momentum=0, decay=0.001, nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimize,
                      metrics=['accuracy'])
        print model.summary()
        history = model.fit(X, y, nb_epoch=1000, batch_size=32, verbose=0)
        plt.figure(1)
        plt.plot(history.history['loss'], label='learning_rate={0}'.format(lr))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y = data()
    train(X, y)
