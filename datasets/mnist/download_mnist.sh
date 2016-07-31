
if ! [ -e train-images-idx3-ubyte.gz]
    then
        wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi

gzip -d train-images-idx3-ubyte.gz

if ! [ -e train-labels-idx1-ubyte.gz ]
	then
		wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi

gzip -d train-labels-idx1-ubyte.gz

if ! [ -e t10k-images-idx3-ubyte.gz ]
	then
		wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d t10k-images-idx3-ubyte.gz

if ! [ -e t10k-labels-idx1-ubyte.gz ]
	then
		wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d t10k-labels-idx1-ubyte.gz