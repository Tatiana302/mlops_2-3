import pandas as pd
import numpy as np
import os
import pickle
from catboost import CatBoostClassifier

import mlflow
from mlflow.tracking import MlflowClient

 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")


train_df = pd.read_csv('/home/deploy/mlops_2-3/datasets/data_train.csv', sep=',')

X = train_df.drop("outcome", axis = 1)
y = train_df['outcome']

n_estimators = 300

cbc = CatBoostClassifier(verbose=0, n_estimators=n_estimators)

with mlflow.start_run():
    mlflow.log_param("number estimators", n_estimators)
    mlflow.log_artifact(local_path="/home/deploy/mlops_2-3/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()


cbc.fit(X, y)

with open("/home/deploy/mlops_2-3/models/model.pkl", "wb") as f:
    pickle.dump(cbc, f)
