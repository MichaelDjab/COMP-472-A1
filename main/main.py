import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier


import matplotlib
from matplotlib import pyplot as plt

with open("goemotions.json", "r") as f:
    data = json.load(f)


post=[] #post list
emotion=[] #emotion list
sentiment =[] #sentiment list
#extracting post, emotion, and sentiment
for elm in data:
    post.append(elm[0])
    emotion.append(elm[1])
    sentiment.append(elm[2])

#for Part 1
plt.title("emotion and sentiment")
plt.xlabel("emotion")
plt.ylabel("sentiment")
plt.legend(["emotion", "sentiment"])
plt.plot(emotion, sentiment)
plt.show()

#Part 2
vectorizer = CountVectorizer()

# Emotion and Sentiment do not need to be vectorized but it's nice to see how it's vectorized
"""
Emo = vectorizer.fit_transform(emotion)
print(vectorizer.get_feature_names_out())
print("The number of tokens for emotion (the size of the vocabulary) : " + str(len(vectorizer.get_feature_names_out())))
print(Emo.toarray())


Sen = vectorizer.fit_transform(sentiment)
print(vectorizer.get_feature_names_out())
print("The number of tokens for sentiment (the size of the vocabulary) : " + str(len(vectorizer.get_feature_names_out())))
print(Sen.toarray())
"""

P= vectorizer.fit_transform(post) #assign number to each word (fitting)

#print(vectorizer.get_feature_names_out()) #printing out all the features
#print(vectorizer.get_feature_names_out()) #printing out only few of the features

#part2.1 display the number of tokens in the dataset
print("The number of tokens for sentiment (the size of the vocabulary) : " + str(len(vectorizer.get_feature_names_out())))

#part2.2
#test_size is 20% so training size is 80%
X_train, X_test, y1_train, y1_test = train_test_split(P, emotion, test_size=0.2) #split for emotion
X_train, X_test, y2_train, y2_test = train_test_split(P, sentiment, test_size=0.2) #split for sentiment

#part 2.3.1
classifier_MNB = MultinomialNB()

model1 = classifier_MNB.fit(X_train, y1_train)#emotion
model2 = classifier_MNB.fit(X_train, y2_train)#sentiment

print("===========MNB============================================")
MNB_test_emo = vectorizer.transform(np.array(["Shut up 15 year-old"]))

MNB_predict_emo = model2.predict(MNB_test_emo)
print('Predicted class = ', MNB_predict_emo)

MNB_test_sen = vectorizer.transform(np.array(["Shut up 15 year-old"]))
#print("Test email = ", MNB_test_sen.toarray())

MNB_predict_sen = model2.predict(MNB_test_sen)
print('Predicted class = ', MNB_predict_sen)


#part 2.3.2
print("=====================DT============================")
dtc = tree.DecisionTreeClassifier()
d_model1 = dtc.fit(X_train, y1_train)

dt_predict1 = d_model1.predict(vectorizer.transform(np.array(["Shut up 15 year-old"])))
print('DT_Predicted class = ', dt_predict1)

#for some reason, MLP is extremely slow so I commented out MLP part for now
print("=====================MLP============================")
#mlp = MLPClassifier().fit(X_train, y1_train)
#print("MLP predicted class = ", mlp.predict(vectorizer.transform(np.array(["Shut up 15 year-old"]))))
