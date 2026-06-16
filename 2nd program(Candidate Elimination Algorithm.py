import pandas as pd

data = pd.read_csv("dataset1.csv")

features = data[['Weather','Temp']].values
labels = data['Result'].values

specific = features[labels=="Yes"][0]

for f in features[labels=="Yes"]:
specific = [specific[i] if specific[i]==f[i] else "?"
for i in range(len(f))]

print("Specific Hypothesis:", specific)

print("General Hypothesis:")
for f in features[labels=="No"]:
print([specific[i] if f[i]!=specific[i] else "?"
for i in range(len(f))])
