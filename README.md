# My first neural network

This is my implementation of the neural network as described in the book [Neural Networks and Deep Learning](https://neuralnetworksanddeeplearning.com) by Michael Nielsen.

The original implementation in the book is in Python, so I thought I would write it in Kotlin, both to force me to really think about the code I was writing, and to see whether Kotlin could be used in place of Python (or Matlab/Octave) for this kind of applications. Koma library was quite helpful, providing matrices operations and plotting utilities.

The network is the one presented in chapters 1 and 2 of the book. I plan to:
1) Comment and polish the code to make it easier to understand, and have it work as a "summary" for the book's content.
2) Improve the network using information from the remaining chapters of the book.
3) Provide some charts and additional data about the results.

For now, the network can reach an accuracy rate of 95.61% after 30 epochs of training over the entire 60k images training set grouped in 10-cases mini-batches, with a learning rate of 3.0, with a single hidden layer of 30 neurons.
