# generate-ann.py
# generates an artificial neural network with random weights
# and an arbitrary number of hidden layers of an arbitrary 
# number of nodes

import sys

if len(sys.argv) < 2:
  print('Usage:')
  print('python generate-ann.py I [ H1 H2 .. Hn ] O')
  print('  I is number of nodes in input layer,')
  print('  Hx is number of nodes in [ optional ] hidden layer x, and')
  print('  O is number of nodes in output layer.')
  exit()

import random

print(' '.join(sys.argv[1:]))
layerNodeCounts = list(map(int, sys.argv[1:]))

for layerIndex in range(1, len(layerNodeCounts)):
  for a in range(layerNodeCounts[layerIndex]):
    for b in range(layerNodeCounts[layerIndex - 1] + 1):
      randomFloat = '%0.3f' % random.uniform(0, 1)
      endCharacter = ' ' if b != layerNodeCounts[layerIndex - 1] else ''
      print(randomFloat, end = endCharacter)
    print()
