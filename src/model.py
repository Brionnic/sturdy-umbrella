import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn stuff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression

# model functionality and validation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# regressor models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class AmesModel():

    def __init__(self):
        # self.model_1 = LinearRegression()

        # only works with model 29a
        # wow, stage 29 model (with ~315 columns) scored
        # median ~0.1328, mean ~0.1398
        #
        # C=0.5 median ~0.1410, mean ~0.1446
        # C=0.75 median ~0.1425, mean ~0.1467
        # C=1.0 median ~0.1350, mean ~0.1386, min ~0.1073
        # C-1.25 median ~0.1331, mean ~0.1448
        # C=1.5 median ~0.1287, mean ~0.1366
        # C=1.75 median ~0.1372, mean ~0.1421
        # C=2.0 median ~0.1364, mean ~0.1398
        # C=4.0 median ~0.1367, mean ~0.1408, min ~0.1034
        # C=8.0 median ~0.1382, mean ~0.1434
        # self.model_1 = SVR(kernel="rbf", C=1.5, gamma='auto')
        # self.model_1 = SVR()

        # Lasso?
        # full auto 0.4034
        # self.model_1 = Lasso(alpha=10)

        # Ridge
        # full auto 0.1473
        # alpha=1: 0.1412, 0.1458
        # alpha=10: 0.1421, 0.1438
        # alpha=50: 0.1458, 0.1456
        # alpha=75: 0.1427, 0.1438
        # alpha=85: 0.1413, 0.1404
        # alpha=87.5: 0.1464, 0.1491
        # alpha=90: 0.1399, 0.1405
        # alpha=0.95: 0.1456, 0.1481
        # alpha=100: 0.1403, 0.1411
        # alpha=125: 0.1447, 0.1456
        # alpha=200: 0.1499, 0.1492
        # alpha=1000: 0.1664, 0.1664
        # self.model_1 = Ridge(alpha=87.5)

        # random forest on stage 29 model (with ~315 columns) scored
        # median ~0.1564, mean ~0.1584 with n_estimators = 100
        # random forest on stage 29 model (with ~315 columns) scored
        # median ~0.1510, mean ~0.1569 with n_estimators = 1000
        # random forest on stage 29 model (with ~315 columns) scored
        # median ~0.1485, mean ~0.1549 with n_estimators = 10000
        # self.model_1 = RandomForestRegressor(n_estimators=100, n_jobs=-1)

        self.model_1 = GradientBoostingRegressor(learning_rate=0.001, n_estimators=10000)

    def fit(self, X, y):
        self.model_1.fit(X, y)

    def transform(self, X_pred):
        self.model_1_preds = self.model_1.predict(X_pred)
        return self.model_1_preds

    # def coefs_(self):
    #     print self.model_1.coef_
