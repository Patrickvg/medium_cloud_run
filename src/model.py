import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./src/weight-height.csv')

dataset['Gender'] = dataset['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

X = dataset.iloc[:, :2]
y = dataset.iloc[:, -1]

regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))
