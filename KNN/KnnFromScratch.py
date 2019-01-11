# Importing Libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

# Importing data
df = pd.read_csv("Iris.csv")
df = df.drop(columns=['Id'], axis=1)

# Partition into features and target
df_feature = df.iloc[:, df.columns != 'Species'].values
df_target = df.iloc[:, df.columns == 'Species'].values


def Accuracy_Score(pred_target, test_target):
    if (len(pred_target) != len(test_target)):
        raise ValueError("pred_target and test_target must be of same length")
    else:
        count = 0
        for i in range(len(test_target)):
            count += (test_target[i] == pred_target[i])
        return (count/len(test_target))


def EuclideanDistance(train_feature, test_features):
    return np.sqrt(np.square(train_feature - test_features))


class KNeighborsClassifier:

    def __init__(self, features, target, K=3):
        self.features = features
        self.target = target
        self.K = K
        self.distance = []

    def featureScaling(self):
        pass
    def calculateDistance(self, test_features):
        test_feature_distance = []
        for i in range(len(self.features)):
            dist = np.sqrt(
                np.sum(np.square(self.features[i, :] - test_features)))
            test_feature_distance.append([dist, self.target[i, 0]])

        test_feature_distance = sorted(test_feature_distance)
        labels = []
        for i in range(self.K):
            labels.append(test_feature_distance[i][1])
        return (Counter(labels).most_common(1)[0][0])

    def predict(self, test_features):
        pred_target = []
        for i in range(len(test_features)):
            pred_target.append(self.calculateDistance(test_features[i, :]))
        return pred_target


# Splitting into training and testing
train_features, test_features, train_target, test_target = train_test_split(
    df_feature, df_target, train_size=0.8, test_size=0.2)

KNN = KNeighborsClassifier(train_features, train_target)
pred_target = KNN.predict(test_features)

print(Accuracy_Score(pred_target, test_target))
