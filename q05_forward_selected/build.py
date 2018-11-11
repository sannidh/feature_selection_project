# %load q05_forward_selected/build.py
# Default imports
from greyatomlib.feature_selection.q05_forward_selected.build import forward_selected
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data/house_prices_multivariate.csv')

def foward_selected(df, model=LinearRegression()):
    X = df.drop(df.columns[len(df.columns)-1], axis=1)
    y = df.iloc[:,-1]
    col_list = X.columns
    def select_cols(df, col_list):
        return df[[item for item in col_list]]
    Variable_f, Variable_f_max = [], []
    Score_r2, Score_r2_max  = [], []
    # the first feature
    for index in range(len(col_list)):
        feature_list=[col_list[index]]
        new_X = select_cols(X,feature_list)
        model.fit(new_X,y)
        r2 = model.score(new_X,y)
        Variable_f.append(new_X.columns[0])
        Score_r2.append(r2)
    max_r2_score = max(Score_r2)
    Variable_f_max = [Variable_f[Score_r2.index(max_r2_score)]]
    Score_r2_max.append(max_r2_score)
    #The next features
    new_Variable_f_max, new_Variable_f, new_Score_r2, new_Score_r2_max, new_Variable_f_max_capture, new_Score_r2_max_capture = [], [], [], [], [], []
    new_Score_r2_max_capture.append(max_r2_score)
    no_features_f_max = len(new_Variable_f)
    new_Variable_f = Variable_f_max
    def features(j):
        feature_list=col_list[j]
        return feature_list
    for index in range(len(col_list)-1):
        for k in range(len(col_list)-1):
            if features(k) not in new_Variable_f:
                new_Variable_f = Variable_f_max.copy()
                new_Variable_f.append(features(k))
                new_X1 = select_cols(X,new_Variable_f)
                model.fit(new_X1,y)
                new_r2 = model.score(new_X1,y)
                new_Score_r2.append(new_r2)
                new_max_r2_score = max(new_Score_r2)
                new_Variable_f_max.append(new_X1.columns)
                largest_indice = new_Score_r2.index(new_max_r2_score)
                new_Variable_f_max_capture =list(new_Variable_f_max[largest_indice])        
        new_Score_r2_max_capture.append(new_max_r2_score)
        Variable_f_max = new_Variable_f_max_capture.copy()
    return Variable_f_max, new_Score_r2_max_capture
foward_selected(data, LinearRegression())




