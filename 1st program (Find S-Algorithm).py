import pandas as pd

data = pd.read_csv("data.csv")

h = ['0', '0']

for _, row in data.iterrows():

    if row['Result'] == 'Yes':

        if h == ['0', '0']:
            h = [row['Weather'], row['Temp']]

        else:
            if h[0] != row['Weather']:
                h[0] = '?'

            if h[1] != row['Temp']:
                h[1] = '?'

print("Hypothesis:", h)
