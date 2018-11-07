# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(df):
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    clf = RandomForestClassifier(random_state=9)
    clf.fit(X, y)
    selector = SelectFromModel(clf, prefit=True)
    selector.transform(X)
    idx_selected = selector.get_support(indices=True)
    features_train = df.iloc[:,idx_selected]
    return list(features_train.columns.values)





