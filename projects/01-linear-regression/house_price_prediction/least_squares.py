import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target

def linear_regression(X, y):
    X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    return theta_best

theta = linear_regression(X[:100], y[:100])
print("coefficients：", theta)