import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("dataset2.csv")

X = data[['Hours']]
y = data['Marks']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

plt.scatter(X, y, label='Data Points')
plt.plot(X, pred, 'r', label='Regression Line')
plt.title("Regression Line")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()

residuals = y - pred

plt.scatter(pred, residuals, label='Residuals')
plt.axhline(y=0, color='r', label='Zero Error Line')
plt.title("Residual Plot")
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals")
plt.legend()
plt.show()

print("MAE =", mean_absolute_error(y, pred))
print("MSE =", mean_squared_error(y, pred))
print("RMSE =", np.sqrt(mean_squared_error(y, pred)))
print("R2 Score =", r2_score(y, pred))
