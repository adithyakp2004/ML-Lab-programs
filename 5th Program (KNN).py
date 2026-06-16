import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("dataset2.csv")

X = data[['Hours', 'Marks']]
y = data['Result']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

passed = data[data['Result'] == 'Yes']
failed = data[data['Result'] == 'No']

plt.scatter(passed['Hours'], passed['Marks'],
            color='green', label='Pass')

plt.scatter(failed['Hours'], failed['Marks'],
            color='red', label='Fail')

plt.scatter(4, 65,
            marker='*', s=200,
            color='blue', label='New Sample')

plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()

result = model.predict([[4, 65]])
print("Prediction:", result[0])
