import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

X = data[['Hours','Marks']]

model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

labels = model.labels_
centroids = model.cluster_centers_

plt.scatter(X['Hours'], X['Marks'],
            c=labels, s=100)

plt.scatter(centroids[:,0], centroids[:,1],
            marker='X', color='red',
            s=200, label='Centroids')

plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.title("K-Means Clustering")
plt.show()
