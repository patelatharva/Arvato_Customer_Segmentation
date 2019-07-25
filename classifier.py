import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import operator
import time
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA    
from sklearn.preprocessing import LabelEncoder
# !pip install mca
# import mca
import chardet
# magic word for producing visualizations in notebook
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
from sklearn.metrics import f1_score
azdias = pd.read_csv('Udacity_AZDIAS_052018.csv', sep=';', dtype=str)
training = pd.read_csv("Udacity_MAILOUT_052018_TRAIN.csv", sep=";", dtype="str")
azdias_cleaned = azdias.copy()
feat_info = pd.read_csv("features.csv")

for index, row in feat_info.iterrows():
    attribute, information_level, var_type, missing, comment = row
    if attribute in azdias_cleaned.columns:
        values = missing.replace("[","").replace("]","").split(",")
        replacement = {}
        for value in values:
            value = value.strip()
            replacement[value] = None
        azdias_cleaned.loc[:, attribute].replace(replacement, inplace=True)

def clean_sl_data(df, azdias_cleaned=azdias_cleaned):
    df = df.copy()
    cat_cols = []
    num_cols = []
    for index, row in feat_info.iterrows():
        attribute, information_level, var_type, missing, comment = row
        if var_type in ["interval", "categorical"]:
            cat_cols.append(attribute)
        elif var_type in ["ordinal", "numeric"]:
            num_cols.append(attribute)
        if attribute in df.columns:
            values = missing.replace("[","").replace("]","").split(",")
            replacement = {}
            for value in values:
                value = value.strip()
                replacement[value] = None
            df.loc[:, attribute].replace(replacement, inplace=True)
#     df.replace({"-1": None, 'X': None, 'XX': None}, inplace=True)
    recode = ['D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',
       'D19_BANKEN_ONLINE_DATUM', 'D19_GESAMT_DATUM',
       'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM',
       'D19_TELKO_DATUM', 'D19_TELKO_OFFLINE_DATUM',
       'D19_TELKO_ONLINE_DATUM', 'D19_VERSAND_DATUM',
       'D19_VERSAND_OFFLINE_DATUM', 'D19_VERSAND_ONLINE_DATUM',
       'D19_VERSI_DATUM', 'D19_VERSI_OFFLINE_DATUM',
       'D19_VERSI_ONLINE_DATUM']
    df[recode] = df[recode].replace("10", "0")
    
#     df_for_cls = df[common_cols].copy()

    to_drop = ["LNR", 'AGER_TYP', 'ALTER_HH', 'ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3',
       'ALTER_KIND4', 'EXTSEL992', 'GEBURTSJAHR', 'HH_DELTA_FLAG',
       'KBA05_BAUMAX', 'KK_KUNDENTYP', 'KKK', 'REGIOTYP', 'TITEL_KZ', 'CAMEO_DEU_2015',
       'W_KEIT_KIND_HH']
    df_for_cls = df.drop(columns=[col for col in to_drop if col in df.columns])
    df_for_cls = df_for_cls[[col for col in azdias_cleaned.columns if col in df_for_cls.columns]]
    for col in df_for_cls.columns:
        df_for_cls.loc[:, col] = df_for_cls[col].fillna(value=azdias_cleaned[col].mode()[0])   
    numbers = [str(x) for x in range(100)]
    
    for col in df_for_cls.columns:
        level = 0
        for value in df_for_cls[col].unique():
            if value not in numbers:
                df_for_cls.loc[df_for_cls[col] == value, col] = level
#                 print(col + " " + str(df_for_cls[col].unique()))                
#                 print("Replaced {} with {}".format(value, level))
                level += 1
        
#     num_cols = [col for col in num_cols if col in common_cols]
#     cat_cols = [col for col in cat_cols if col in common_cols]
#     df_for_cls[num_cols] = df_for_cls[num_cols].astype(float)
#     for col in cat_cols:
#         df_for_cls = pd.concat([df_for_cls.drop(columns=[col]), pd.get_dummies(df_for_cls[col].astype(str))], axis=1)
#     pd.get_dummies(df_for_cls[cat_cols]) = df_for_cls[cat_cols].astype(str)
    df_for_cls = df_for_cls.astype(float)
    return df_for_cls

from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return clean_sl_data(df=input_array)

X = training.drop(columns=["RESPONSE"])
Y = training["RESPONSE"].astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=42)

lgbm_clf = lgb.LGBMClassifier(objective='binary', metric='auc')
lgbm_clf.get_params()

clf_pipeline = Pipeline([   
            ("cleaner", DataCleaner()),
            ("clf", lgbm_clf)
    
])
param_grid = {'clf__learning_rate': [0.01],
              'clf__num_iterations': [200],
              'clf__boosting_type': ['gbdt'],
              'clf__num_leaves': [62],
              'clf__random_state': [42]}
cv_pipeline = GridSearchCV(estimator=clf_pipeline, param_grid=param_grid, scoring='roc_auc', cv=2)
# cv_pipeline = clf_pipeline

cv_pipeline.fit(X_train, Y_train)

print(cv_pipeline.best_score_)
print(cv_pipeline.best_estimator_)

Y_pred = cv_pipeline.predict(X_test)
print(pd.DataFrame(Y_pred).loc[:, 0].value_counts())
print(Y_test.value_counts())
print(f1_score(y_true=Y_test, y_pred=Y_pred))
Y_pred_proba = cv_pipeline.predict_proba(X_test)                 
auc = roc_auc_score(Y_test, Y_pred_proba[:, 1])
print(auc)