import pandas as pd
import numpy as np
from time import time
import os, sys
import glob
from xgboost import XGBClassifier

select_features = {"raw":{  "model1":"features_raw_fea_S_%s_1.csv",
                            "model2":"features_raw_fea_S_%s_2.csv",
                            "model3":"features_raw_fea_S_%s_3.csv",
                            "model7":"features_raw_fea_S_%s_7.csv",
                            
                         }
                  }

df = {
    "train":pd.read_csv("train.csv",header=None,names=["Id",'class']).set_index("Id").sort_index(),
    "val": pd.read_csv("val.csv",header=None,names=["Id",'class']).set_index("Id").sort_index(),
}

def feature_label(feature="top",dataset = "train"):
    print(feature)
    dfs=[]
    for key in sorted(select_features[feature].keys()):
        df_this = pd.read_csv(select_features[feature][key] % (dataset)).set_index("Id").sort_index() #.astype('category')
        columns = [col+"_%s" % (key) for col in df_this.columns] 
        df_this.columns = columns
        
        dfs.append(df_this[columns]) 
    if dataset=='test':
        df_f_l = pd.concat(dfs,axis=1)
        df_f_l["class"] = -1
    else:
        dfs.append(df[dataset])
        df_f_l = pd.concat(dfs,axis=1)
        
    return df_f_l.iloc[:,:-1], df_f_l.iloc[:,[-1]]

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

scoring = make_scorer(accuracy_score)

#tuning model

from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from scipy.stats import uniform
from xgboost import XGBRegressor

import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization

# Load the diabetes dataset (for regression)
X,y = feature_label(feature="raw",dataset = "val")

# Instantiate an XGBRegressor with default hyperparameter settings
clf = XGBClassifier()


bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.09)},
       {'name': 'n_estimators', 'type': 'discrete', 'domain': (500, 1000)},
       {"name": "max_depth", 'type':'discrete', 'domain': (3, 10)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 5)},
         {'name': 'reg_alpha', 'type': 'continuous', 'domain': (0, 0.05)},
         {'name': 'reg_lambda', 'type': 'continuous', 'domain': (0.1, 1)},
         {'name': 'subsample', 'type': 'continuous', 'domain': (0.1, 0.6)},
         {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.1, 0.6)},
        ]

# Optimization objective 
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                    XGBClassifier(learning_rate=parameters[0],\
                        n_estimators=int(parameters[1]),\
                        max_depth=int(parameters[2]),\
                        min_child_weight=int(parameters[3]),\
                        reg_alpha = parameters[4],\
                        reg_lambda = parameters[5],\
                        subsample = parameters[6],\
                        colsample_bytree = parameters[7],\
                        n_jobs=5,\
                        objective= "multi:softprob",\
                        colsample_bylevel = 1,\
                        booster="gbtree",\
                        scale_pos_weight = 1,\
                        gamma = 0,\
                       ), 
                X, y, scoring="accuracy", cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=100),n_jobs=10).mean()
    score = np.array(score)
    return score

optimizer = BayesianOptimization(f=cv_score, 
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True, 
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=50)


from xgboost import XGBClassifier
X1,y1 = feature_label(feature="raw",dataset = "val")
X2,y2 = feature_label(feature="raw",dataset = "train")
X = pd.concat([X1,X2])
y = pd.concat([y1,y2])
np.random.seed(100)
re_sample = {1:2, 2: 1.7, 3:1.7, 4:1.5, 5:1.5, 6:1.55, 7:1.5, 8:0.6, 9:2, 10:2}
indexes = []
for m_class in re_sample:
    index_m_class = y[y['class']==m_class].index
    replace = True
    if re_sample[m_class]<1:
        replace = False
    indexes.append(np.random.choice(index_m_class ,\
                                    int(len(index_m_class)*re_sample[m_class]), replace=replace))

select = np.concatenate(indexes)

X = X.loc[select,:]
y = y.loc[select,:]

parameters = optimizer.X[np.argsort(optimizer.Y,axis=0).reshape(-1)][0]
clf_xgb = XGBClassifier(learning_rate=parameters[0],\
                        n_estimators=int(parameters[1]),\
                        max_depth=int(parameters[2]),\
                        min_child_weight=int(parameters[3]),\
                        reg_alpha = parameters[4],\
                        reg_lambda = parameters[5],\
                        subsample = parameters[6],\
                        colsample_bytree = parameters[7],\
                        n_jobs=5,\
                        objective= "multi:softprob",\
                        colsample_bylevel = 1,\
                        booster="gbtree",\
                        scale_pos_weight = 1,\
                        gamma = 0,\
                       )

clf_xgb.fit(X,y)




'''
# can enable and check this parameter"

from xgboost import XGBClassifier
X1,y1 = feature_label(feature="raw",dataset = "val")
X2,y2 = feature_label(feature="raw",dataset = "train")
X = pd.concat([X1,X2])
y = pd.concat([y1,y2])
np.random.seed(100)
re_sample = {1:2, 2: 1.7, 3:1.7, 4:1.5, 5:1.5, 6:1.55, 7:1.5, 8:0.6, 9:2, 10:2}
indexes = []
for m_class in re_sample:
    index_m_class = y[y['class']==m_class].index
    replace = True
    if re_sample[m_class]<1:
        replace = False
    indexes.append(np.random.choice(index_m_class ,\
                                    int(len(index_m_class)*re_sample[m_class]), replace=replace))

select = np.concatenate(indexes)

X = X.loc[select,:]
y = y.loc[select,:]

clf_xgb = XGBClassifier(learning_rate=0.05,\
                    n_estimators=700,\
                    min_child_weight=1,\
                    booster="gbtree",\
                    max_depth=5,\
                    scale_pos_weight = 1,\
                    reg_alpha = 0.001,\
                    reg_lambda = 0.3,\
                    subsample = 0.5,\
                    colsample_bytree = 0.3,\
                    n_jobs=5,\
                    objective= "multi:softmax",
                   )

clf_xgb.fit(X,y)
'''


from sklearn.externals import joblib

joblib.dump(clf_xgb, "xgb_search.sav")
