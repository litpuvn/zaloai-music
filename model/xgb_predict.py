import argparse
import pandas as pd
import numpy as np
from sklearn.externals import joblib
parser = argparse.ArgumentParser(description='up sampling')

parser.add_argument('--type', default="test", type=str, help = "which type of test")

args = parser.parse_args()

select_features = {                           
                   "raw":{  "model1":"features_raw_fea_S_%s_1.csv",
                            "model2":"features_raw_fea_S_%s_2.csv",
                            "model3":"features_raw_fea_S_%s_3.csv",
                            "model7":"features_raw_fea_S_%s_7.csv",
                         }
                  }

df = {
    "test":pd.read_csv("test.csv",header=None,names=['Id']).set_index("Id").sort_index(),
 
}


def feature_label(feature="top",dataset = "train"):
    print(feature)
    dfs=[]
    for key in sorted(select_features[feature].keys()):
        df_this = pd.read_csv(select_features[feature][key] % (dataset)).set_index("Id").sort_index() #.astype('category')
        columns = [col+"_%s" % (key) for col in df_this.columns] 
        df_this.columns = columns
        
        columns = [col for col in columns\
                   if col.split("_")[-2]]
        
        dfs.append(df_this[columns]) 
    if dataset=='test':
        df_f_l = pd.concat(dfs,axis=1)
        df_f_l["class"] = -1
    else:
        dfs.append(df[dataset])
        df_f_l = pd.concat(dfs,axis=1)
        
    return df_f_l.iloc[:,:-1], df_f_l.iloc[:,[-1]]



model = joblib.load("xgb_search.sav")

X_test,_ = feature_label(feature="raw",dataset = args.type) #privatetest

df_submit = pd.DataFrame({"Id":X_test.index.values,"Genre":model.predict(X_test)}).set_index("Id").reset_index()
df_submit.to_csv("/result/submission.csv",header=True,index=False)
