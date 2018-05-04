# coding: utf-8
# this program building different ML models for predict user's  credits
# we need to build a function only for accessing an individual's credit
import pandas as pd
import sklearn, time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Multinomial Naive Bayes Classifier

def naive_bayes(x_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(x_train, y_train)
    return model


# KNN classifier

def knn(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    return model


# logistic Regression classifier
def logistic_regression(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(x_train, y_train)
    return model


# Random Forest Classifier
def random_forest(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(x_train, y_train)
    return model


# Decision Tree Classifier
def decision_tree(x_train, y_train):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting(x_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(x_train, y_train)
    return model


# SVM Classifier
def svm(x_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(x_train, y_train)
    return model


# NN classifier
def neural_network(x_train, y_train):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #     model = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
    model.fit(x_train, y_train)
    return model


df = pd.read_csv('/home/wilson/Desktop/databases/db_labelled.csv')
df.drop(['user_nickname', 'content', 'signature'], axis=1, inplace=True)
df.dropna(subset=['LABEL'], inplace=True)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
x_train = train_set.drop(['count_pu', 'count_dg', 'count_zh', 'LABEL'], axis=1)
y_train = train_set['LABEL'].copy()
x_test = test_set.drop(['count_pu', 'count_dg', 'count_zh', 'LABEL'], axis=1)
y_test = test_set['LABEL'].copy()
classifiers_list = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT', 'NN']
classifiers = {'NB': naive_bayes,
               'KNN': knn,
               'LR': logistic_regression,
               'RF': random_forest,
               'DT': decision_tree,
               'SVM': svm,
               'GBDT': gradient_boosting,
               'NN': neural_network,
               }

for classifier in classifiers_list:
    start_time = time.time()
    print("=============%s==============" % classifier)
    model = classifiers[classifier](x_train, y_train)
    end_time = time.time()
    print("training time is: %s" % (end_time - start_time))
    predict = model.predict(x_test)
    precision = metrics.precision_score(y_test, predict)
    print('precision is %s' % precision)
    recall = metrics.recall_score(y_test, predict)
    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy is: %s' % accuracy)

