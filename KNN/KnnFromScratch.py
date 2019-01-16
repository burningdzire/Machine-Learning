# Importing Libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Importing Data
df = pd.read_csv('Iris.csv', index_col='Id')
df.index.name = None

# Partition into attributes and target
X = df.loc[:, df.columns != 'Species']
Y = df['Species']


# def Accurary_Score(pred_target, test_target):
#     if (len(pred_target) != len(test_target)):
#         return -1
#     else:
#         count = 0
#         for i in range(len(pred_target)):
#             if(pred_target[i] == test_target[i]):
#                 count += 1

#         return count/len(pred_target)


class KNeighborsClassifier:
    def __init__(self, train_attributes, train_target, K=5):
        self.train_attributes = train_attributes
        self.train_target = train_target
        self.K = K

    def calculateTarget(self, test_attribute_index, test_attribute):
        distance = []
        for train_attributes_index, train_attribute in self.train_attributes.iterrows():
            dist = np.sqrt(np.sum(np.square(train_attribute - test_attribute)))
            distance.append([dist, self.train_target[train_attributes_index]])

        distance = pd.Series(distance)
        distance = distance.sort_values(ascending=True)
        print(distance)
        # label = []
        # for i in range(self.K):
        #     label.append(distance[i][1])
        # label = pd.Series(label)
        # return label.value_counts().head(1)

    def predict(self, test_attributes, test_target):
        pred_target = []
        for test_attribute_index, test_attribute in test_attributes.iterrows():
            # pred_target.append(self.calculateTarget(test_attribute_index, test_attribute))
            self.calculateTarget(test_attribute_index, test_attribute)
            break

        # print(pred_target)
        # print(test_target)
        # return Accurary_Score(pred_target, test_target)


train_attributes, test_attributes, train_target, test_target = train_test_split(
    X, Y, train_size=0.8, test_size=0.2)
KNN = KNeighborsClassifier(train_attributes, train_target)
KNN.predict(test_attributes, test_target)
