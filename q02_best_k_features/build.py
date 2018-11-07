# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile, f_regression, SelectFromModel

def percentile_k_features(df, k=20):
    selector = SelectPercentile(f_regression, percentile=k)
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    selector.fit(X,y)
    idx_selected = selector.get_support(indices=True)
    idx_sorted = [idx_selected for _, idx_selected in sorted(zip(selector.scores_[idx_selected], idx_selected), reverse=True)]
    features_train = df.iloc[:,idx_sorted]
    return list(features_train.columns.values)



