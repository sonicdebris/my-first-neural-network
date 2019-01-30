curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output src/main/resources/train-images.gz
gunzip src/main/resources/train-images.gz

curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output src/main/resources/train-labels.gz
gunzip src/main/resources/train-labels.gz

curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --output src/main/resources/test-images.gz
gunzip src/main/resources/test-images.gz

curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --output src/main/resources/test-labels.gz
gunzip src/main/resources/test-labels.gz
