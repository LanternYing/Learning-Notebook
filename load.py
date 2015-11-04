import numpy as np
import struct
import os

data_predir = './datasets/'


def one_hot(x, n):
    '''
    x: the label matrix
    n: total type of labels
    this function change the label into this form : 0 0 0 0 0 0 0 1 0 0 (8)
    '''
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    print x.dtype
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(binary=True):
    data_dir = os.path.join(data_predir, 'mnist/')
    fin = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    magic, ntrain, nrow, ncol = struct.unpack('>4I', fin.read(16))
    trainX = np.fromfile(fin, dtype=np.uint8).reshape((ntrain, nrow*ncol)).astype(float)
    # print "trainX: magic = {0}, num = {1}, nrow = {2}, ncol = {3}".format(magic, num, nrow, ncol)
    # print "trainX shape:" + str(trainX.shape)

    fin = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    magic, num = struct.unpack('>II', fin.read(8))
    trainY = np.fromfile(fin, dtype=np.uint8).reshape((num))

    # print "trainY: magic = {0}, num = {1}".format(magic, num)
    # print "trainY shape:" + str(trainY.shape)

    fin = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    magic, ntest, nrow, ncol = struct.unpack('>4I', fin.read(16))
    testX = np.fromfile(fin, dtype=np.uint8).reshape((ntest, nrow*ncol)).astype(float)

    # print "testX: magic = {0}, num = {1}, nrow = {2}, ncol = {3}".format(magic, num, nrow, ncol)
    # print "testX shape:" + str(testX.shape)

    fin = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    magic, num = struct.unpack('>II', fin.read(8))
    testY = np.fromfile(fin, dtype=np.uint8).reshape((num))

    # print "testY: magic = {0}, num = {1}".format(magic, num)
    # print "testY shape:" + str(testY.shape)

    def show():
        from matplotlib import pyplot
        import matplotlib as mpl
        pyplot.ion()
        for i in xrange(ntrain):
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
            imgplot = ax.imshow(trainX[i, :].reshape((28, 28)), cmap=mpl.cm.Greys)
            imgplot.set_interpolation('nearest')
            ax.set_title(str(trainY[i]), fontsize=20)
            # ax.xaxis.set_ticks_position('top')
            # ax.yaxis.set_ticks_position('left')
            pyplot.show()
            _ = raw_input("Press [enter] to continue. [c + enter] to close. ")
            if _ == 'c':
                break
            pyplot.close()
    # show()

    trainX = trainX / 255.0
    testX = testX / 255.0

    if binary:
        trainY = one_hot(trainY, 10)
        testY = one_hot(testY, 10)
    else:
        trainY = np.asarray(trainY)
        testY = np.asarray(testY)
        # print testY.shape

    return trainX, trainY, testX, testY



