import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("dataset2.csv")

X = data[["Hours", "Marks"]]
y = data["Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = MLPClassifier(hidden_layer_sizes=(4,),
                      activation='logistic',
                      solver='sgd',
                      learning_rate_init=0.1,
                      max_iter=5000,
                      random_state=42)

model.fit(X, y_encoded)

pred = model.predict(X)

print("Inputs:")
print(X)
print("\nActual Output:")
print(le.inverse_transform(y_encoded))
print("\nPredicted Output:")
print(le.inverse_transform(pred))
