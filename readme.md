# neural network

An implementation of a highly configurable artificial neural network.

## Description

This is an implementation of a highly configurable artificial neural network.  Artificial neural networks of arbitrary size can be constructed, trained on training data, and tested on testing data, with good results.  I.e., these neural network constructions can contain any number of hidden layers, and any given layer (input, hidden, or output) can contain any number of nodes.

`train.out` trains an untrained artificial neural network using specified training data.

`test.out` tests a trained artificial neural network using specified testing data.

`utils/generate-ann.py` generates weights for an arbitrarily-sized artificial neural network.

`utils/format-data.py` generates training and testing data from CSV data.

All artificial neural network (ANN) files, training data files, and testing data files must adhere to a strict structure, each of which is described below.

### ANN file structure

For an ANN with an input layer, n hidden layers, and an output layer, the ANN file is structured as follows:
```
I H1 H2 .. Hn O
1{0,1} 1{1,1} .. 1{I,1}
1{0,2} 1{1,2} .. 1{I,2}
..
1{0,H1} 1{1,H1} .. 1{I,H1}
..
x{0,1} x{1,1} .. x{H(x-1),1}
x{0,2} x{1,2} .. x{H(x-1),2}
..
x{0,Hx} x{1,Hx} .. x{H(x-1),Hx}
..
O{0,1} O{1,1} .. O{Hn,1}
O{0,2} O{1,2} .. O{Hn,2}
..
O{0,O} O{1,O} .. O{Hn,O}
```
where `I` is the number of nodes in the input layer, `Hx` is the number of hnodes in hidden layer `x`, and `O` is the number of nodes in the output layer.  The following lines specify weights from nodes of the previous layer to nodes of the current layer.  For example, `x{p,c}` is the weight from the `p`th node of the previous layer `x-1` to the `c`th node of the current layer `x`.  Weights are represented as floating point values.  Every layer has 0th node which is used as a negative bias weight, so each line should have a total of `t+1` weights, where `t` is the number of nodes in the previous layer.

As an example, take an ANN file representing an ANN with 5 input nodes, 4 nodes in the first hidden layer, 3 nodes in the second hidden layer, and 2 nodes in the output layer:
```
5 4 3 2
0.735 0.939 0.019 0.556 0.720 0.465
0.021 0.088 0.682 0.069 0.823 0.538
0.239 0.486 0.857 0.891 0.174 0.292
0.587 0.462 0.070 0.985 0.647 0.644
0.468 0.854 0.379 0.383 0.904
0.052 0.440 0.473 0.208 0.484
0.959 0.846 0.826 0.559 0.161
0.429 0.017 0.376 0.075
0.481 0.664 0.984 0.132
```

## Dependencies

* Python 2.7 (for `utils/`)
* gcc (with C++11 support)

## Build

To build:
```
make
```

## Usage

To run `train.out`:
```
./train.out
```

To run `test.out`:
```
./test.out
```
