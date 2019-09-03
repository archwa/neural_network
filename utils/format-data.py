# format-data.py
# used to format csv data into a usable train/test structure

import sys

if len(sys.argv) < 2:
  print('Usage:')
  print('python format-data.py <file> <fraction>')
  print('  <file> is the path of a csv data file')
  print('  <fraction> is the fraction of data used for training; the remainder of data is used for testing')
  exit()

import pandas as pd
import numpy as np

filePath = sys.argv[1]
fraction = float(sys.argv[2])

# import the data
raw_df = pd.read_csv(filePath, header = None)

# TODO: feature extraction using covariance or other metric
# drop features with no data
df = raw_df.dropna(axis = 1, how = "all")

# drop all data points with any empty values
df = df.dropna()

# encode categorical columns
for index in range(0, len(df.columns)):
	if any(type(n) is str for n in df[index]):
		df[index] = pd.Categorical(df[index]) # change to categorical type
		df[index] = df[index].cat.codes # encode the categories

# normalize the columns and eliminate useless features
for index in range(0, len(df.columns)):
	columnMax = (df[index].abs()).max() 
	if columnMax == 0:
		del df[index]
	else:
		df[index] = df[index].divide(columnMax)

# extract training and testing data sets randomly
train = df.sample(frac = fraction).round(decimals = 3)
test = df.drop(train.index).round(decimals = 3)

# write training and testing data sets to file
train.to_csv("train.csv", float_format = "%.3f", sep = " ", header = False, index = False)
test.to_csv("test.csv", float_format = "%.3f", sep = " ", header = False, index = False)
