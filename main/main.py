import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import classification_report


with open("goemotions.json", "r") as f:
    dataset = json.load(f)


def write_classification(X_test, y_test, model, model_name, parameters) :
    f.write(model_name + " model with " + parameters + '\n')
    f.write(model_name + " confusion matrix" + '\n')
    f.write(np.array2string(metrics.confusion_matrix(y_test, model.predict(X_test))) + '\n')
    f.write(model_name + " classification report"+ '\n')
    f.write(classification_report(y_test, model.predict(X_test)) + '\n')


def print_classification(X_test, y_test, classifier, model_name, parameters) :
    print(model_name + " model with " + parameters)
    print(model_name + " confusion matrix")
    print(metrics.confusion_matrix(y_test, classifier.predict(X_test)))
    print(model_name + " classification report")
    print(classification_report(y_test, classifier.predict(X_test)))




comments = []
emotions = []
sentiments = []

for triplet in dataset:
    comments.append(triplet[0])
    emotions.append(triplet[1])
    sentiments.append(triplet[2])


# Plot the distribution of the posts in each category

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

# show histogram
#plt.show()

plt.title("Distribution of Sentiments")
sentiments_count = Counter(sentiments)
sentiments_labels, sentiments_values = zip(*sentiments_count.items())

sentiments_values_ind_in_desc_order = np.argsort(sentiments_values)[::-1]
#rearrange data

sentiments_labels = np.array(sentiments_labels)[sentiments_values_ind_in_desc_order]
sentiments_values = np.array(sentiments_values)[sentiments_values_ind_in_desc_order]
sentiment_indexes = np.arange(len(sentiments_labels))
plt.bar(sentiment_indexes, sentiments_values)

plt.xticks(sentiment_indexes, sentiments_labels)

plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.3)

# show histogram
#plt.show()


#writing to a file
f = open ("performance.txt", "w") #overwrite


#Part 2
vectorizer = CountVectorizer()

P= vectorizer.fit_transform(comments) #assign number to each word (fitting)


#part2.1 display the number of tokens in the dataset
print("The number of tokens for sentiment (the size of the vocabulary) : " + str(len(vectorizer.get_feature_names_out())))


#part2.2

X_train1, X_test1, y1_train, y1_test = train_test_split(P, emotions, test_size=0.2) #split for emotion
X_train2, X_test2, y2_train, y2_test = train_test_split(P, sentiments, test_size=0.2) #split for sentiment

#part 2.3.1
classifier_MNB = MultinomialNB()

#model1 = classifier_MNB.fit(X_train1, y1_train)#emotion

#Base-MNB emotions
classifier_MNB.fit(X_train1, y1_train)#emotion
print_classification(X_test1, y1_test, classifier_MNB, "Base-MNB emotions", "default parameters")
write_classification(X_test1, y1_test, classifier_MNB, "Base-MNB emotions", "default parameters")

#model2 = classifier_MNB.fit(X_train2, y2_train)#sentiment

#Base-MNB sentiments
classifier_MNB.fit(X_train2, y2_train)#sentiment
print_classification(X_test2, y2_test, classifier_MNB, "Base-MNB sentiments", "default parameters")
write_classification(X_test2, y2_test, classifier_MNB, "Base-MNB sentiments", "default parameters")




#MNB_predict_sen = model2.predict(MNB_test_sen)
#print('Predicted class = ', MNB_predict_sen)



#part 2.3.2
#DT
dtc = tree.DecisionTreeClassifier()




dtc.fit(X_train1, y1_train)
print_classification(X_test1, y1_test, dtc, "Base-DT emotions", "default parameters")
write_classification(X_test1, y1_test, dtc, "Base-DT emotions", "default parameters")
dtc.fit(X_train2, y2_train)
print_classification(X_test2, y2_test, dtc, "Base-DT sentiments", "default parameters")
write_classification(X_test2, y2_test, dtc, "Base-DT sentiments", "default parameters")





#Base-MLP
#mlp_model1 = MLPClassifier(max_iter=1).fit(X_train1, y1_train)
#mlp_model2 = MLPClassifier(max_iter=1).fit(X_train2, y2_train)
mlp = MLPClassifier(max_iter=1).fit(X_train1, y1_train)
print_classification(X_test1, y1_test, mlp, "Base-MLP emotions", "default parameters")
write_classification(X_test1, y1_test, mlp, "Base-MLP emotions", "default parameters")


mlp = MLPClassifier(max_iter=1).fit(X_train2, y2_train)
print_classification(X_test2, y2_test, mlp, "Base-MLP sentiments", "default parameters")
write_classification(X_test2, y2_test, mlp, "Base-MLP sentiments", "default parameters")

#TOP-MNP

search_space = {
    "alpha" : [0, 0.5, 1.5, 2.0]
}


GS = GridSearchCV(estimator=MultinomialNB(),
                  param_grid = search_space)
#GS_model1 = GS.fit(X_train1, y1_train)
#GS_model2 = GS.fit(X_train2, y2_train)




GS.fit(X_train1, y1_train)
print_classification(X_test1, y1_test, GS, "Top-MNB emotions", "alpha : [0, 0.5, 4, 5]")
write_classification(X_test1, y1_test, GS, "Top-MNB emotions", "alpha : [0, 0.5, 4, 5]")

GS.fit(X_train2, y2_train)
print_classification(X_test2, y2_test, GS, "Top-MNB sentiments", "alpha : [0, 0.5, 4, 5]")
write_classification(X_test2, y2_test, GS, "Top-MNB sentiments", "alpha : [0, 0.5, 4, 5]")



#TOP-DT
search_space_DT = {
    "criterion" : np.array(["entropy"]),
    "max_depth" : np.array([10, 11]),
    'min_samples_split' : np.array([5, 10, 15])
}

GS_DT = GridSearchCV(tree.DecisionTreeClassifier(),
                     search_space_DT)
#GS_DT_model1 = GS_DT.fit(X_train1, y1_train)
#GS_DT_model2 = GS_DT.fit(X_train2, y2_train)

GS_DT.fit(X_train1, y1_train)
print_classification(X_test1, y1_test, GS_DT, "Top-DT emotions", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")
write_classification(X_test1, y1_test, GS_DT, "Top-DT emotions", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")

GS_DT.fit(X_train2, y2_train)
print_classification(X_test2, y2_test, GS_DT, "Top-DT sentiments", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")
write_classification(X_test2, y2_test, GS_DT, "Top-DT sentiments", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")



#Top-MLP
search_space_Top_MLP = {
    'activation' : np.array(['identity', 'logistic', 'tanh', 'relu']),
    'hidden_layer_sizes' : [(30, 50), (10, 10, 10)],
    'solver' : ['adam', 'sgd']
}

GS_MLP = GridSearchCV(MLPClassifier(max_iter=1),
                      search_space_Top_MLP,
                      )
#GS_MLP_model1 = GS_MLP.fit(X_train1, y1_train)
#GS_MLP_model2 = GS_MLP.fit(X_train2, y2_train)
GS_MLP.fit(X_train1, y1_train)
print_classification(X_test1, y1_test, GS_MLP, "Top-MLP emotions", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")
write_classification(X_test1, y1_test, GS_MLP, "Top-MLP emotions", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")

GS_MLP.fit(X_train2, y2_train)
print_classification(X_test2, y2_test, GS_MLP, "Top-MLP sentiments", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")
write_classification(X_test2, y2_test, GS_MLP, "Top-MLP sentiments", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")




#part 2.5

X_train3, X_test3, y3_train, y3_test = train_test_split(P, emotions, test_size=0.7) #split for emotion
X_train4, X_test4, y4_train, y4_test = train_test_split(P, sentiments, test_size=0.7) #split for sentiment



#Base-MNB emotions
classifier_MNB.fit(X_train3,  y3_train)#emotion
print_classification(X_test3, y3_test, classifier_MNB, "Base-MNB emotions", "default parameters")
write_classification(X_test3, y3_test, classifier_MNB, "Base-MNB emotions", "default parameters")
#Base-MNB sentiments
classifier_MNB.fit(X_train4, y4_train)#sentiment
print_classification(X_test4, y4_test, classifier_MNB, "Base-MNB sentiments", "default parameters")
write_classification(X_test4, y4_test, classifier_MNB, "Base-MNB sentiments", "default parameters")




#MNB_predict_sen = model2.predict(MNB_test_sen)
#print('Predicted class = ', MNB_predict_sen)



#part 2.3.2
#DT


dtc.fit(X_train3, y3_train)
print_classification(X_test3, y3_test, dtc, "Base-DT emotions", "default parameters")
write_classification(X_test3, y3_test, dtc, "Base-DT emotions", "default parameters")

dtc.fit(X_train4, y4_train)
print_classification(X_test4, y4_test, dtc, "Base-DT sentiments", "default parameters")
write_classification(X_test4, y4_test, dtc, "Base-DT sentiments", "default parameters")





#Base-MLP
mlp_model3 = MLPClassifier(max_iter=1).fit(X_train3, y3_train)
mlp_model4 = MLPClassifier(max_iter=1).fit(X_train4, y4_train)


print_classification(X_test3, y3_test, mlp_model3, "Base-MLP emotions", "default parameters")
write_classification(X_test3, y3_test, mlp_model3, "Base-MLP emotions", "default parameters")

print_classification(X_test4, y4_test, mlp_model4, "Base-MLP sentiments", "default parameters")
write_classification(X_test4, y4_test, mlp_model4, "Base-MLP sentiments", "default parameters")

#TOP-MNP




GS.fit(X_train3, y3_train)
print_classification(X_test3, y3_test, GS, "Top-MNB emotions", "alpha : [0, 0.5, 4, 5]")
write_classification(X_test3, y3_test, GS, "Top-MNB emotions", "alpha : [0, 0.5, 4, 5]")

GS.fit(X_train4, y4_train)
print_classification(X_test4, y4_test, GS, "Top-MNB sentiments", "alpha : [0, 0.5, 4, 5]")
write_classification(X_test4, y4_test, GS, "Top-MNB sentiments", "alpha : [0, 0.5, 4, 5]")




#TOP-DT


GS_DT.fit(X_train3, y3_train)
print_classification(X_test3, y3_test, GS_DT, "Top-DT emotions", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")
write_classification(X_test3, y3_test, GS_DT, "Top-DT emotions", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")

GS_DT.fit(X_train4, y4_train)
print_classification(X_test4, y4_test, GS_DT, "Top-DT sentiments", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")
write_classification(X_test4, y4_test, GS_DT, "Top-DT sentiments", "criterion: entropy, max_depth: 10, 11, min_sample: 5, 10, 15")



#Top-MLP
GS_MLP.fit(X_train3, y3_train)
print_classification(X_test3, y3_test, GS_MLP, "Top-MLP emotions", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")
write_classification(X_test3, y3_test, GS_MLP, "Top-MLP emotions", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")

GS_MLP.fit(X_train4, y4_train)
print_classification(X_test4, y4_test, GS_MLP, "Top-MLP sentiments", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")
write_classification(X_test4, y4_test, GS_MLP, "Top-MLP sentiments", "activation: sigmoid, tanh, relu, identity, solver: adam and sgd")



f.close()
#Part 2.5
# with test size of 70%, it significantly decreases the accuracy score for all the 6 classifiers, it is because only 30% of dataset was used to train so more datasets of training lead better overall accuracy score.

