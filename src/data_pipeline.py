import pandas as pd
import numpy as np

class GetData():
    def __init__(self):
        # don't really do anything here
        # but should have

    def fit(self, X, y):
        self.X = X
        self.y = y

    def transform(self):
        # transform to log prices
        self.y = np.log(y)

    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform()
