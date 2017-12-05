from sklearn import datasets
from collections import Counter
import numpy as np
from sklearn.cross_validation import train_test_split


class knn():
    
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            label = self.closest(x)
            predictions.append(label)
        self.predictions = predictions
        return predictions

    def closest(self, x_test):
        best_dist = []
        best_index = []
        for i in range(self.k):
            best_dist.append(np.linalg.norm(x_test - self.x_train[i]))
            best_index.append(i)

        for i in range(len(self.y_train)):
            dist = np.linalg.norm(x_test - self.x_train[i])
            for j in range(self.k):
                if dist < best_dist[j]:
                    best_dist[j] = dist
                    best_index[j] = self.y_train[i]
                    break

        win = Counter(best_index).most_common(1)[0][0]
        return win

    def score(self,y_test):
        s = np.sum(np.array([self.predictions==y_test]))
        return (s/len(y_test))*100

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train ,y_test = train_test_split(x,y, test_size = 0.5)

clf = knn()
clf.fit(x_train,y_train)
clf.predict(x_test)
print(clf.score(y_test))
