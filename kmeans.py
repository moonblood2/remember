from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class kmeans():
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train):

        centroids = {}
        for i in range(self.k):
            centroids[i]=x_train[i]


        for i in range(300):
            classifications = {}
            for i in range(self.k):
                classifications[i]=[]

            for x in x_train:
                dist = [np.linalg.norm(x-centroids[centroid]) for centroid in centroids]
                classification = dist.index(np.min(dist))
                classifications[classification].append(x)

            for classification in classifications:
                centroids[classification] = np.mean(classifications[classification], axis = 0)

        return centroids

iris = datasets.load_iris()

##x = iris.data
x = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [7,6],
              [1.9,1]])


clf = kmeans(k=2)
d = clf.fit(x)
plt.scatter(x[:,0],x[:,1])
for i in d:
    plt.scatter(d[i][0],d[i][1], marker='*')
plt.show()
