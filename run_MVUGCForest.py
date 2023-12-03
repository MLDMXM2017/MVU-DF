'''
Description: 在单折数据上训练一个MVUGCForest
Author: tanqiong
Date: 2023-05-14 11:03:42
LastEditTime: 2023-09-01 14:18:21
LastEditors: tanqiong
'''
from MVUGCForest.MVUGCForest import MVUGCForest
from MVUGCForest.evaluation import accuracy,f1_binary,f1_macro,f1_micro, mse_loss, aupr, auroc
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import time

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

import os

import shutil
from preprocessing_data import  split_multiview_data

def get_config():
    config={}
    
    # boost_features
    config["is_stacking_for_boost_features"] = False
    config["boost_feature_type"] = 'opinion'            # 'opinion', 'probability', None
    
    # high order features
    config["is_RIT"] = True
    config["is_intersection_inner_view"] = True
    config["is_intersection_cross_views"] = True
    config["enforce_intra"] = True                        # 是否强迫生成intra-section feature
    config["intra_feature_least"] = 1                     # 强迫生成intra-section的数量
    config["inter_method"] = "operator"                   # 'stack' or 'operator'
    config["filter_intra"] = False
    config["filter_inter"] = False
   
    # cluster samples
    config["span"]=2
    config["cluster_method"] = "uncertainty_bins"                     # 'uncertainty', 'span', 'uncertainty_bins'
    config["n_bins"] = 30
    config["ta"] = 0.7
    
    # resample
    config["is_resample"] = True              # True/Fasle, conflict with 'is_resample'
    config["onn"] = 3                       # outlier nearest neighbour
    config["layer_resample_method"] = "integration"    # "integration", "respective"
    config["accumulation"] = False

    # training 
    config["max_layers"]=2
    config["early_stop_rounds"]=1  
    config["is_weight_bootstrap"] = True
    config["train_evaluation"]=accuracy    # accuracy, f1_macro, aupr, auroc, mse_loss
    config["view_opinion_generation_method"] = "joint"  # 'mean'/'joint'/'sum' (sum==mean)
    config["is_save_model"] = False
    config["random_state"]=666     # 669172976

    # prediction
    config["is_des"] = False
    config["use_layerwise_opinion"] = True              # 是否考虑层间联合opinion

    # node configs
    config["uncertainty_basis"] = "evidence"    # "entropy"/"evidence"
    config["evidence_type"] = "probability"     # "knn" / "probability" knn舍弃
    config["act_func"] = "approx_step"                 # 'approx_step', 'ReLU', None
    config["W_type"] = "sum"               # 'n_class', 'n_tree', 'sum', 'variable'
    config["use_kde"] = False               # 是否使用kde以额外考虑数据不确定度
    config["estimator_configs"]=[]
    for _ in range(2):
        config["estimator_configs"].append({"n_fold":5, 
                                            "type": "RandomForestClassifier",
                                            ### sklearn parameters ###
                                            "n_estimators": 20, 
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 2,
                                            })
    for _ in range(2):
        config["estimator_configs"].append({"n_fold": 5, 
                                            "type": "ExtraTreesClassifier",
                                            ### sklearn parameters ###
                                            "n_estimators": 20, 
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 2, 
                                            })
    return config

if __name__=="__main__":
    # load datasets
    features_dict = {}
    feature_names = ["Gene Expression.csv" ,"Morgan Fingerprint.csv", "Pubchem Fingerprint.csv", "Drug Targets.csv"]
    for i, filename in enumerate(feature_names):
        features_dict[i] = pd.read_csv(f"./data/features/{filename}", index_col=0).values
    labels = pd.read_csv("./data/labels.csv", index_col=0).values

    x_train, x_test, y_train, y_test = split_multiview_data(features_dict, labels)
    print(len(y_test))
    config=get_config()
    # config['logger_path'] = "MVUGCForest_info"
    
    # train
    gc=MVUGCForest(config)
    gc.fit_multiviews(x_train,y_train, evaluate_set="all")
    # predict
    y_proba, opinion = gc.predict_opinion(x_test, y_test, is_record=True, bypass=True)

    y_test_pred = np.argmax(y_proba,axis=1)
    print(f"\ntest acc: {accuracy(y_test, y_proba)}\ttest f1: {f1_macro(y_test, y_proba)}")