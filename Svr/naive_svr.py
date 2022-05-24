import os
import sys
import warnings
from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR,LinearSVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import  make_pipeline
from sklearn.compose import make_column_transformer


warnings.filterwarnings('ignore')
random_seed = 9999

# load data
train_df = pd.read_csv(r'D:\Envision Racing\data\train.csv')
test_df = pd.read_csv(r'D:\Envision Racing\data\test.csv')

# display(train_df.head())
# display(test_df.head())

# modify column names
def mod_names(df):
    df.columns = df.columns.str.lstrip() 
    df.columns = df.columns.str.lower()

# drop unwanted features
def drop_cols(df):
    df.drop(columns = {
        "number",
        "driver_number",
        "lap_improvement",
        "s1",
        "s2",
        "s3",
        "s1_improvement",
        "s2_improvement",
        "s3_improvement",
        "crossing_finish_line_in_pit",
        "pit_time",
        "group",
        "team",
        "power",
        "kph",
        "hour",
        "elapsed"
    },inplace=True)

# drop rows with missing values
# nan_ft = ['s3_large','s1_large','s2_large']
def drop_na_rows(df):
    df.fillna(0, inplace=True)
    # df.dropna(subset=nan_ft, axis=0, inplace=True)

# convert hour feature to seconds
def convert_time_to_seconds(x):
    if type(x) is (int or float):
        return x
    else:
        min = x.split(":")[0]
        sec = x.split(":")[1]
        sec = float(sec)
        min = float(min)
        sec = sec/60
        return min + sec

# cleanse data
def cleanse(df):
    mod_names(df)
    drop_cols(df)
    drop_na_rows(df)
    df["s3_large"] = df["s3_large"].apply(convert_time_to_seconds)
    df["s2_large"] = df["s2_large"].apply(convert_time_to_seconds)
    df["s1_large"] = df["s1_large"].apply(convert_time_to_seconds)

cleanse(train_df)
cleanse(test_df)

# split data in to train and val set
X_train,X_val,Y_train,Y_val = train_test_split(train_df.drop(columns=["lap_time"]),
                                               train_df["lap_time"],
                                               test_size=0.05,
                                               random_state=random_seed,
                                               stratify=train_df['location'])


cat_fts = ["driver_name","location","event"]
num_fts = ["lap_number","s1_large","s2_large","s3_large"]

ohe = OneHotEncoder(sparse=False)
std_scaler = StandardScaler()

# transformers = [
#     (std_scaler, num_fts),
#     (ohe, cat_fts)
# ]
ct = make_column_transformer(
                        (std_scaler, num_fts),
                        (ohe, cat_fts),
                        remainder='passthrough',
                        n_jobs=-1,
                        verbose=True)
# svr model
naive_svr = LinearSVR(verbose=True)


naive_svr_pipeline = make_pipeline(ct,naive_svr)
print("model is training.............")

# cross validate svr model
cv_scores = cross_val_score(
    naive_svr_pipeline,X_train,Y_train,cv=5,
    scoring="neg_mean_squared_log_error",
    verbose=True)

print("CV scores:",cv_scores)
print("mean cv score:",np.mean(cv_scores))

naive_svr_pipeline.fit(X_train,Y_train)
print("model is trained.............")

print("Training score: ",(naive_svr_pipeline.score(X_train,Y_train)))

# make predictions
y_pred = naive_svr_pipeline.predict(X_val)
dump(naive_svr_pipeline,r'naive_svr_ppline.joblib')

print("validation score: ",np.sqrt(metrics.mean_squared_log_error(Y_val,y_pred)))
print("Validation_fit score: ",(naive_svr_pipeline.score(X_val,Y_val)))

# load model
naive_svr_pipeline = load(r'naive_svr_ppline.joblib')
# make predictions
preds = naive_svr_pipeline.predict(test_df.drop(columns=["lap_time"]))

submission = pd.read_csv(r'data\submission.csv')
submission['LAP_TIME'] = preds
submission.to_csv('naive_svr_submission_file.csv', index=False)

print(submission.head())
print(submission.describe())


# with open('naive_svr_output.txt',"a+") as f:
#     print("CV scores:",cv_scores,file= f)
#     print("mean cv score:",np.mean(cv_scores),file= f)
#     print("Training score: ",np.sqrt(naive_svr_pipeline.score(X_train,Y_train)),file= f)
#     print("validation score: ",np.sqrt(metrics.mean_squared_log_error(Y_val,y_pred)),file=f)
#     print("Validation_fit score: ",np.sqrt(naive_svr_pipeline.score(X_val,Y_val)),file=f)
#     print("============================================")