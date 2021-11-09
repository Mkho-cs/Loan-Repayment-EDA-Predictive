from typing import List
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,roc_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt

"""
Data preprocessing
"""
def create_cat_encoder(cols_cat: list)->ColumnTransformer:
    cat_encoder = ColumnTransformer([("ordinal_enc", OrdinalEncoder(), cols_cat)], remainder='passthrough')
    return cat_encoder

def create_label_encoder(target: list)->ColumnTransformer:
    return ColumnTransformer([('label_enc', LabelEncoder(), target)], remainder='passthrough')

def create_pipeline(pipe: list)->Pipeline:
    pipeline = Pipeline(pipe)
    return pipeline

def pipeline_fit_transform(pipeline: Pipeline, data):
    return pipeline.fit_transform(data)

def create_minmax_scaler():
    return MinMaxScaler()

def create_standard_scaler(cols_cont: list)->ColumnTransformer:
    return ColumnTransformer([("standard_scaler", StandardScaler(), cols_cont)], remainder='passthrough')

"""
Predictive modelling
"""

def evaluate_roc_auc(model, train: dict, val: dict, fit=False):
    if fit: model.fit(train[0], train[1])

    #y_train_pred = model.predict(train[0])
    y_train_proba = model.predict_proba(train[0])[:, 1]

    #y_val_pred = model.predict(val[0])
    y_val_proba = model.predict_proba(val[0])[:, 1]

    train_auc_roc = roc_auc_score(train[1], y_train_proba)
    val_auc_roc = roc_auc_score(val[1], y_val_proba)

    return train_auc_roc, val_auc_roc

def plot_roc(model, val:dict, title="Classifier"):
    y_proba = model.predict_proba(val[0])[:, 1]
    fpr, tpr, _ = roc_curve(val[1], y_proba)
    plt.plot(fpr, tpr)
    plt.plot((0, 1), ls='dashed', color='red')

    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.show()



    

