import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Read dataset
df = pd.read_csv('./data.csv')

# Create training set and test set
y = df['diagnosis'].replace(['M', 'B'], [1, 0]).values
x = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=.7, random_state=100)

# Normalize training set
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Create logistic regression model
model = LogisticRegression(penalty='l2', dual=False, max_iter=5000)

# fit data
model.fit(scaled_X_train, y_train)

# predict and compute accuracy
y_pred = model.predict(scaled_X_test)
score = accuracy_score(y_test, y_pred, normalize=True)
