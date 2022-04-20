import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Finding the path to file
for dirpath, _, file_names in os.walk("data/"):
    for filename in file_names:
        print(os.path.join(dirpath, filename))


training = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# print(training.head())

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training,test])

# print(all_data.columns)

# Take a closer look at the data given
print(training.info())
print(training.describe())