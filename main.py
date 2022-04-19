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