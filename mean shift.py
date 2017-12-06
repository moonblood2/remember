from sklearn import datasets
import numpy as np
import pdb
import matplotlib.pyplot as plt

class meanshift():

    def __init__(self, r=4):
        self.r = r
        
    def fit(self, data):
        centroids={}
        for i in range(len(data)):
            centroids[i] = data[i]

        for i in range(300):
            new_centroids=[]

            for centroid in centroids:
                within_r = [x for x in data if np.linalg.norm(centroids[centroid]-x)<self.r]
                mean_centroid = np.mean(within_r, axis=0)
                new_centroids.append(tuple(mean_centroid))

            unique = list(set(new_centroids))
            
            centroids = {}
            for i in range(len(unique)):
                centroids[i] = unique[i]

        self.centroids = centroids

            

iris = datasets.load_iris()
x=iris.data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

clf = meanshift()
clf.fit(X)
print(clf.centroids)

plt.scatter(X[:,0],X[:,1])
centroids = clf.centroids
for centroid in centroids:
    plt.scatter(centroids[centroid][0],centroids[centroid][1])

plt.show()
