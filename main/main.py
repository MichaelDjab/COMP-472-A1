import json
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import tree
# from sklearn.neural_network import MLPClassifier

# import matplotlib
#


# 1. Dataset Preparation & Analysis (5pts)

# 1.1 - 1.2 Download & load dataset
with open("goemotions.json", "r") as f:
    dataset = json.load(f)

# 1.4 Extract posts and the two labels plot the distribution of the posts in each category

comments = []
emotions = []
sentiments = []

for triplet in dataset:
    comments.append(triplet[0])
    emotions.append(triplet[1])
    sentiments.append(triplet[2])

# Plot the distribution of emotions in each category
emotions_plot = plt.figure(1)
plt.title("Distribution of Emotions")
emotions_count = Counter(emotions)
emotions_labels, emotions_values = zip(*emotions_count.items())

# sort in descending order
emotion_values_ind_in_desc_order = np.argsort(emotions_values)[::-1]

# rearrange data
emotions_labels = np.array(emotions_labels)[emotion_values_ind_in_desc_order]
emotions_values = np.array(emotions_values)[emotion_values_ind_in_desc_order]
emotion_indexes = np.arange(len(emotions_labels))
plt.bar(emotion_indexes, emotions_values)

# add labels
plt.xticks(emotion_indexes, emotions_labels)

# adjustments
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)

# Plot the sentiments of the posts in each category
sentiments_plot = plt.figure(2)
plt.title("Distribution of Sentiments")
sentiments_count = Counter(sentiments)
sentiments_labels, sentiments_values = zip(*sentiments_count.items())

# sort in descending order
sentiments_values_ind_in_desc_order = np.argsort(sentiments_values)[::-1]

# rearrange data
sentiments_labels = np.array(sentiments_labels)[sentiments_values_ind_in_desc_order]
sentiments_values = np.array(sentiments_values)[sentiments_values_ind_in_desc_order]
sentiments_indexes = np.arange(len(sentiments_labels))
plt.bar(sentiments_indexes, sentiments_values)

# add labels
plt.xticks(sentiments_indexes, sentiments_labels)

# adjustments
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)

# save figures
emotions_plot.savefig('emotions_plot.pdf')
sentiments_plot.savefig('sentiments_plot.pdf')