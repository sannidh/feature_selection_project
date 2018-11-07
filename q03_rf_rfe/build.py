# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(df):
    model = RandomForestClassifier()
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    selector = RFE(model, int(df.shape[1]/2))
    selector.fit(X,y)
    idx_selected = selector.get_support(indices=True)
    idx_sorted = [idx_selected for _, idx_selected in sorted(zip(selector.ranking_[idx_selected],idx_selected))]
    features_train = df.iloc[:,idx_sorted]
    return list(features_train.columns.values)





