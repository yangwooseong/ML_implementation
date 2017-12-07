import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/train.csv', header=None)

def get_summary(x, y):
    # split train by class
    train_byclass = dict()
    # extract unique values of labels
    values = np.unique(y.values)

    for value in values:
        train_byclass[value] = pd.concat([x[y == value], y[y == value]], axis=1).iloc[:, :-1]

    # get mean and deviation by features
    summary = dict()
    for label, df in train_byclass.items():
        tmp = dict()
        for col in list(df.columns):
            tmp[col] = [df[col].mean(), df[col].std()]
        summary[label] = tmp
    return(summary)

# probability of a sample when the popluation is Gaussian
def prob(x, mean, std):
    exp = math.exp(- (x-mean)**2 / (2*std*std))
    return exp/(math.sqrt(2 * math.pi) * std)

# probability of class for each sample
def prob_xi(summary, x_i, y):
    # x_i : pd.Series
    labels = [0, 1]
    probs = dict()

    for label in labels:
        probs[label] = 1
        for idx, val in x_i.iteritems():
            probs[label] *= prob(val, summary[label][idx][0], summary[label][idx][1])
        # prior
        prob_label = (y.values == label).sum() / y.shape[0]
        probs[label] *= prob_label
    return probs

# prediction
def predict(probs):
    # probs : dict of probability of class
    return max(probs, key=probs.get)

# get prediction of all the samples
def get_predictions(summary, x_train, y_train):
    pre = pd.Series([-1], index=x_train.index)
    for idx, row in x_train.iterrows():
        pre[idx] = predict(prob_xi(summary, row, y_train))
    return pre

# get accuracy
def accuracy(y, pred):
    y.sort_index(inplace=True)
    pred.sort_index(inplace=True)
    return (y==pred).sum() / y.shape[0]

# split data into train data and test data
train = train.sample(frac=1).reset_index(drop=True)
x = train.iloc[:,:-1]
y = train.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    random_state = 1, test_size=0.3, stratify=y)

# fit the parameters of gaussian distribution using train data
train_summary = get_summary(x_train, y_train)

# predict the test data
prediction = get_predictions(train_summary, x_test, y_train)
print("Accuracy of Naive Bayesian Classifier is : %.5f\n(Without cross validation)" %accuracy(y_test, prediction))
