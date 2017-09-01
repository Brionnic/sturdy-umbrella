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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class AmesModel():

    def __init__(self):
        # self.model_1 = LinearRegression()
        self.model_1 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)

    def fit(self, X, y):
        self.model_1.fit(X, y)

    def transform(self, X_pred):
        self.model_1_preds = self.model_1.predict(X_pred)
        return self.model_1_preds
