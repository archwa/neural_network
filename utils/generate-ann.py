#!/usr/bin/python
import random
import sys

print(' '.join(sys.argv[1:]))
layerNodeCounts = list(map(int, sys.argv[1:]))

for layerIndex in range(1, len(layerNodeCounts)):
	for a in range(layerNodeCounts[layerIndex]):
		for b in range(layerNodeCounts[layerIndex - 1] + 1):
			randomFloat = '%0.3f' % random.uniform(0, 1)
			endCharacter = ' ' if b != layerNodeCounts[layerIndex - 1] else ''
			print(randomFloat, end = endCharacter)
		print()
