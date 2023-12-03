from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaseEnsemble
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import euclidean_distances
from typing import Tuple, Dict
from numpy import ndarray as arr
from copy import deepcopy
from .util import accuracy, f1_macro, auroc, aupr
from .logger import get_logger, get_custom_logger
from .uncertainty import *
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import os
'''
此处导入新的包
'''

class NodeClassifier(object):
    def __init__(self, layer_id, view_id, index, config, random_state, logger_path=None):
        self.fitted = False      
        self.config: dict = config
        self.name = "layer_{}, view_{}, estimstor_{}, {}".format(
            layer_id, view_id, index, self.config["type"])
        self.estimator_type: str = self.config["type"]
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = None
        # print("node random_state: ", self.random_state)
        self.n_fold = self.config.get("n_fold", 5)
        self.config.pop("n_fold", None)
        self.estimator_class = globals()[self.config["type"]]
        self.config.pop("type")
        self.uncertainty_basis = self.config.get("uncertainty_basis", "evidence")   # "entropy"/"evidence"
        self.config.pop("uncertainty_basis", None)
        self.evidence_type = self.config.get("evidence_type", "probability")           # "knn" / "probability"
        self.config.pop("evidence_type", None)
        self.des = self.config.get("is_des", False)
        self.config.pop("is_des", None)
        self.n_tree_in_one_forest = self.config.get("n_estimators", 50)
        
        self.act_func = self.config.get('act_func', 'approx_step') # 计算证据的激活函数: 'approx_step', 'ReLU', None
        self.config.pop('act_func', None)
        self.W_type = self.config.get('W_type', 'sum')          # 'n_class', 'n_tree', 'sum', 'variable' 
        self.config.pop('W_type', None)

        self.use_kde = self.config.get('use_kde', False)
        self.config.pop('use_kde', None)

        self.estimators_: Dict[int, BaseEnsemble] = {i:None for i in range(self.n_fold)}
        self.n_class = None
        self.neighs_ = [None for _ in range(self.n_fold)]
        self.kdes_: Dict[int, KernelDensity] = {i:None for i in range(self.n_fold)}
        self.kde_score_mean_in_distribution: List[float] = []     # shape of (#folds, )
        self.train_labels = [None for _ in range(self.n_fold)]
        self.n_feature_origin: int                                  # 原始特征的数量, 用于进行核密度估计
        self.n_sample_origin: int


        # 日志记录
        if logger_path is None:
            self.logger_path = "./MVUGCForest_info"
        else:
            self.logger_path = logger_path
        assert os.path.exists(self.logger_path), f"logger_path: {self.logger_path} not exist! "
        self.LOGGER_2 = get_custom_logger("KFoldWrapper", "Node_train_log.txt", self.logger_path)

    def _init_estimator(self)->RandomForestClassifier:
        """初始化基本森林学习器"""
        if self.estimator_type in ["XGBClassifier", "LGBMClassifier"]:
            assert self.config.get("num_class"), "请为XGBoost分类器设置 'num_class' 参数"
            
        # elif self.estimator_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        estimator_args = self.config
        est_args = estimator_args.copy()
        est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)

    def fit(self, 
            x, y, n_feature_origin:int=None, sample_weight:arr=None, 
            n_sample_origin: int = None,
            ) -> Tuple[arr, arr]:
        """
        
        Returns
        -------
        y_proba_wrapper: ndarray, shape is (n_samples, n_classes)
            验证集概率
        opinion_wrapper: ndarray, shape is (n_samples, n_classes+1)
            结点的opinion
        evidence_wrapper: ndarray or None, ndarray shape is (#samples, n_classes). 
            仅当使用基于证据计算不确定性时, 该返回值才是ndarray, 否则返回None
        """
        self.LOGGER_2.info(f"----------------------------------------------------------------------------------------------")
        skf = StratifiedKFold(n_splits=self.n_fold,
                              shuffle=True, random_state=self.random_state)
        cv = [(t, v) for (t, v) in skf.split(x, y)]

        n_sample = len(y)
        n_class = len(np.unique(y))
        self.n_class = n_class

        wrapper_opinion = np.empty((n_sample, n_class+1))
        wrapper_evidence = np.empty((n_sample, n_class)) if self.uncertainty_basis == "evidence" else None

        wrapper_kde_score = np.empty(n_sample)

        self._fit_estimators(x=x, y=y, cv=cv, sample_weight=sample_weight)
        # self._fit_neighbors(x=x, y=y, cv=cv, r=None)

        # 计算训练阶段的opinions
        for k in range(self.n_fold):
            est = self.estimators_[k]
            train_id, val_id = cv[k]

            if self.estimator_type in ["XGBClassifier", "LGBMClassifier"]:
                # 如果是梯度提升树
                if self.estimator_type == "XGBClassifier":
                    margin_value = est.predict(x[val_id], output_margin=True)
                elif self.estimator_type == "LGBMClassifier":
                    margin_value = est.predict(x[val_id], raw_score=True)
                base_opinion = get_opinion_base_proba_mat(margin_value, est_type=self.estimator_type)
            elif self.estimator_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
                # 如果是并行树
                if self.uncertainty_basis == "evidence":
                    # 使用证据计算并行森林的不确定度
                    if self.evidence_type == "knn":
                        # 基于K近邻计算不确定度
                        evidence = self.__get_evidence_base_knn(x[val_id], forest_id=k) 
                    elif self.evidence_type == "probability":
                        # 基于概率计算不确定度
                        if (self.W_type == 'variable'):
                            evidence, W = self.__get_evidence_base_proba(x[val_id], k, self.act_func, True)
                        evidence = self.__get_evidence_base_proba(x[val_id], k, self.act_func)
                    else:
                        raise Exception("please set true parameter for \'evidence_type\'")
                    
                    # 生成opinion, shape of (#val, #n_class+1)
                    if self.W_type == 'n_class':
                        base_opinion = get_opinion_base_evidence(evidence, W=self.n_class)
                    elif self.W_type == 'n_tree':
                        base_opinion = get_opinion_base_evidence(evidence, W=self.n_tree_in_one_forest)
                    elif self.W_type == 'sum':
                        base_opinion = get_opinion_base_evidence(evidence, W=self.n_tree_in_one_forest+self.n_class)
                    elif self.W_type == 'variable':
                        # 如果是可变的
                        base_opinion = get_opinion_base_evidence(evidence, W=W)

                elif self.uncertainty_basis in ["entropy", None]:
                    # 使用熵计算并行森林的不确定度
                    y_proba_mat = predict_proba_mat_parallel(est, x[val_id])
                    base_opinion = get_opinion_base_proba_mat(y_proba_mat, est_type=self.estimator_type)
                else:
                    assert True, "参数uncertainty_basis必须设置正确"

            wrapper_opinion[val_id] = base_opinion
            y_proba = opinion_to_proba(base_opinion)
            if self.uncertainty_basis == "evidence":
                wrapper_evidence[val_id] = evidence

            self.LOGGER_2.info(
                "{}, n_fold_{}, Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
                self.name, k, accuracy(y[val_id], y_proba), f1_macro(y[val_id], y_proba), 
                auroc(y[val_id], y_proba), aupr(y[val_id], y_proba)))
        
        # 计算训练阶段的kde_score
        if self.use_kde:
            self.n_feature_origin = n_feature_origin
            
            for k in range(self.n_fold):
                train_id, val_id = cv[k]
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x[train_id, :n_feature_origin])
                mean_log_density_train = np.mean( kde.score_samples(x[train_id, :n_feature_origin]) )
                log_density_val = kde.score_samples(x[val_id, :n_feature_origin])
                wrapper_kde_score[val_id] = self._normalize_log_density( 
                    mean_log_density_train, log_density_val 
                )

                self.kdes_[k] = kde
                self.kde_score_mean_in_distribution.append(mean_log_density_train)
            # 将kde_score度融入opinion
            wrapper_opinion[:, -1] += wrapper_kde_score
            wrapper_opinion /= np.sum(wrapper_opinion, axis=1, keepdims=True)

            
        # 计算堆叠的opinion表达的概率
        y_proba_wrapper = opinion_to_proba(wrapper_opinion)

        self.LOGGER_2.info("{}, {},Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
            self.name, "wrapper", accuracy(y, y_proba_wrapper), f1_macro(y,y_proba_wrapper),
            auroc(y, y_proba_wrapper), aupr(y, y_proba_wrapper)))
        self.LOGGER_2.info("----------")

        self.cv = [(t[t<n_sample_origin], v[v<n_sample_origin]) for t, v in cv]
        self.X_train = x[:n_sample_origin]
        self.y_train = y[:n_sample_origin]
        self.fitted = True  # 是否拟合的标志
        return y_proba_wrapper, wrapper_opinion, wrapper_evidence    

    def predict_proba(self, x_test) -> arr:
        proba, _ = self.predict_opinion(x_test)
        return proba

    def predict_opinion(self, x_test)->Tuple[arr, arr]:
        """
        Returns
        -------
        proba: ndarray of shape (#samples, #classes)

        """
        evidence = None
        if self.estimator_type in ["XGBClassifier", "LGBMClassifier"]:
            base_opinions = []
            for est in self.estimators_.values():
                if self.estimator_type == "XGBClassifier":
                    margin_value = est.predict(x_test, output_margin=True)
                elif self.estimator_type == "LGBMClassifier":
                    margin_value = est.predict(x_test, raw_score=True)
                base_opinions.append(
                    get_opinion_base_proba_mat(margin_value, est_type=self.estimator_type))
            # 计算联合opinion  
            # wrapper_opinion = joint_multi_opinion(base_opinions)
            wrapper_opinion = np.mean(base_opinions, axis=0)
            # 计算概率
            proba = opinion_to_proba(wrapper_opinion)

        # 并行森林, 将所有决策树放在一起计算不确定度
        elif self.estimator_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
            if self.uncertainty_basis == "evidence":
                # 基于证据计算不确定度
                evidence = self._get_evidence(x_test, self.evidence_type)

                if self.W_type == 'n_class':
                    wrapper_opinion = get_opinion_base_evidence(evidence, W=self.n_class)
                elif self.W_type == 'n_tree':
                    wrapper_opinion = get_opinion_base_evidence(evidence, W=self.n_tree_in_one_forest)
                elif self.W_type == 'sum':
                    wrapper_opinion = get_opinion_base_evidence(evidence, W=self.n_tree_in_one_forest+self.n_class)
                elif self.W_type == 'variable':
                    # 如果是可变的, evidence是一个tuple, 存储evidence 和 W
                    W = evidence[1]
                    evidence = evidence[0]
                    wrapper_opinion = get_opinion_base_evidence(evidence, W)
                
                proba = opinion_to_proba(wrapper_opinion)

            elif self.uncertainty_basis in ["entropy", None]:
                # 基于熵计算不确定度
                proba_mat = []
                for est in self.estimators_.values():
                    proba_mat.extend(predict_proba_mat_parallel(est, x_test))
                # 计算opinion  
                wrapper_opinion = get_opinion_base_proba_mat(proba_mat, est_type=self.estimator_type)
                # 计算概率
                proba = opinion_to_proba(wrapper_opinion)

        # 计算预测阶段的kde_score, 与证据收集方式保持一致, 对五折kde_score取平均
        wrapper_kde_score = 0
        if self.use_kde:
            for k in range(self.n_fold):
                log_density_test = self.kdes_[k].score_samples(x_test[:,:self.n_feature_origin])
                wrapper_kde_score += self._normalize_log_density( 
                    self.kde_score_mean_in_distribution[k], log_density_test
                )
            wrapper_kde_score /= self.n_fold

            wrapper_opinion[:, -1] += wrapper_kde_score
            wrapper_opinion /= np.sum(wrapper_opinion, axis=1, keepdims=True)

        return proba, wrapper_opinion, evidence
    
    def _fit_estimators(self, x, y, cv, sample_weight=None):
        """拟合基学习器(森林)"""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, val_id = cv[k]
            # print(x[train_id])
            if hasattr(est, "early_stopping_rounds"):
                est.fit(x[train_id], y[train_id],
                        eval_set=[(x[val_id], y[val_id])])
            else:
                est.fit(x[train_id], y[train_id], sample_weight=sample_weight[train_id])
            self.estimators_[k] = est

        
    def _fit_neighbors(self, x, y, cv, r=None):
        """基于预测结果训练为每一个森林训练Nearest Neighbors"""
        
        if r is None:
            if hasattr(self.estimators_[0], "n_estimators"):
                n_tree = self.estimators_[0].n_estimators
            else:
                n_tree = len(self.estimators_[0].estimators_)
            r = np.sqrt(self.n_tree_in_one_forest / 5)
        for i, (train_idx, val_idx) in enumerate(cv):
            prediction_mat = self._apply_tree_prediction(x[val_idx], i)

            # 如果是多分类, 对prediction_mat做onehot编码
            if self.n_class > 2:
                ohe = OneHotEncoder(sparse=False)
                prediction_mat = ohe.fit_transform(prediction_mat)
            neigh = NearestNeighbors(radius=r).fit(prediction_mat)
            self.neighs_[i] = neigh
            self.train_labels[i] = y[val_idx]

    def __get_evidence_base_knn(self, x, forest_id):
        """基于K近邻收集证据, 指定森林的索引, 用于训练阶段, 在验证集上获取证据
        Parameters
        ----------
        x: ndarray, 待预测的样本(验证集)
        forest_id: int, 森林分类器的编号
        """
        def count_labels_nn(mask, y_list):
            """统计经过掩码后的样本标签中各个类别的数量"""
            counts = [np.sum(y_list[mask]==c) for c in np.unique(y_list)]
            return np.array(counts)
        prediction_mat = self._apply_tree_prediction(x, forest_id)
        r = np.mean(euclidean_distances(prediction_mat, prediction_mat))
        nn_gragh = self.neighs_[forest_id].radius_neighbors_graph(prediction_mat, radius=r).toarray()
        nn_gragh = nn_gragh.astype(bool)
        # 计算标签值
        evidence = np.apply_along_axis(count_labels_nn, axis=1, arr=nn_gragh, 
                                       y_list=self.train_labels[forest_id] )
        return evidence

    def __get_evidence_base_proba(self, x, forest_id, func='approx_step', return_W:bool=False):
        """基于预测的概率结果计算证据
        Parameters
        ----------
        x: ndarray, 待预测的样本(验证集)
        forest_id: int, 森林分类器的编号
        
        Returns
        -------
        evidence: ndarray of shape (#samples, #classes)
        """
        from .util import get_des_weights, get_des_weights_fast, record_trees_feature_indices
        proba_mat = self._apply_tree_predict_proba(x, forest_id) # shape=(#samples, #classes*#trees)
        n_class = self.n_class
        # n_est = len(self.estimators_[forest_id].estimators_)
        n_sample = x.shape[0]
        proba_mat = np.split(proba_mat, [n_class*i for i in range(1,self.n_tree_in_one_forest,1)], axis=1)    # shape=(#trees, #samples, #classes)
        proba_mat = np.array(proba_mat)

        # 0623-决策树的权重
        if self.des and self.fitted:
            n_tree = len(self.estimators_[forest_id].estimators_)
            tree_features_list = record_trees_feature_indices(self.estimators_[forest_id])
            y_preds = np.argmax(proba_mat, axis=2).T    # (samples, #trees)

            tree_weights = get_des_weights_fast(x, y_preds, tree_features_list,
                                           x_train=self.X_train[self.cv[forest_id][0]],
                                           y_train=self.y_train[self.cv[forest_id][0]],
            )              # shape = (#samples, #trees)
            tree_weights = n_tree * tree_weights.T # shape = (#tree, #samples), 每一列是一个样本的权重
        else:
            tree_weights = np.ones(proba_mat.shape[:-1])
        
        evidence = np.empty(shape=(n_sample, n_class))

        if func == 'ReLU':
            activate = partial(ReLU)
        elif func == 'approx_step':
            activate = partial(approx_step, 
                               bias = 1/n_class,
                               k = 10,)
        elif func == None:
            activate = lambda x:x
        else:
            raise Exception("activate function error")

        if n_class==2:
            # 二分类
            e_0 = activate(x = proba_mat[:, :, 0] - proba_mat[:, :, 1]) * tree_weights # proba_mat切片的shape是(#trees, #samples), ReLU之后的shape是(#trees, #sample)
            e_1 = activate(x = proba_mat[:, :, 1] - proba_mat[:, :, 0]) * tree_weights

            # 
            e_0 = np.where(e_0>1, 1, e_0)
            e_1 = np.where(e_1>1, 1, e_1)
            evidence[:, 0] = np.sum(e_0, axis=0)                        # sum之后的shape: (#samples, )
            evidence[:, 1] = np.sum(e_1, axis=0)
            if return_W:
                assert(np.all(np.sum(e_0+e_1))<=1), "total evidence must less than 1 when W set to None"
                W = np.sum(1-(e_0+e_1), axis=0) 
        """
        else:
            # 多分类, 暂不支持
            for c in range(n_class):
                mean_proba_other_class = np.mean(np.delete(proba_mat, c, axis=2), 
                                                 axis=2)  # shape: (#trees, #samples)
                evidence[:, c] = np.sum(activate(x = proba_mat[:, :, c] - mean_proba_other_class), 
                                        axis=0)
        """
        if return_W:
            return evidence, W
        return evidence 


    def _get_evidence(self, x, evidence_type:str):
        """收集证据, 在所有森林上收集证据并使用证据之和作为最终证据, 用于预测. 
            使用K近邻的真实标签作为证据 或 使用概率矩阵作为证据来源.
            收集的证据是对五折交叉森林产出证据取平均的结果 

        Parameters
        ----------
        x: ndarray,
            待预测的样本
        evidence_type: str, 'knn' or 'probability', 证据来源的依据

        Returns
        -------
        evidence: ndarray of shape (#samples, #classes)
        """
        evidence = 0
        W = None
        for i, _ in self.estimators_.items():
            if evidence_type == 'knn':
                evidence += self.__get_evidence_base_knn(x, i)
            elif evidence_type == 'probability':
                if (self.W_type == 'variable'):
                    evidence, W = self.__get_evidence_base_proba(x, i, self.act_func, True)
                evidence += self.__get_evidence_base_proba(x, i, self.act_func)
        evidence = evidence / self.n_fold
        if W is not None:
            # 如果W被赋值, 则需要返回W
            return evidence, W
        return evidence
    
    def _apply_tree_prediction(self, x_for_predict, forest_id):
        """针对待预测样本, 获取森林(序号为forest_id)中所有树的预测结果
        
        returns
        -------
        prediction_mat: ndarray of shape (#samples, #trees)"""
        prediction_mat = get_trees_predict(self.estimators_[forest_id], x_for_predict)
        return prediction_mat
    
    def _apply_tree_predict_proba(self, x_for_predict, forest_id):
        """针对待预测样本, 获取森林(序号为forest_id)中所有树的预测结果
        
        returns
        -------
        proba_mat: ndarray of shape (#samples, #trees * #classes)
        """
        proba_mat = get_trees_predict_proba(self.estimators_[forest_id], x_for_predict)
        return proba_mat
    
    def _normalize_log_density(self, x, y):
        return ReLU(x-y)/np.abs(x)
        # return ReLU(x-y)/100



def get_trees_predict(forest, X):
    """获取森林中每一棵树的预测结果
    Parameters
    ----------
    forest: estimator
    X: array_like
    
    Returns
    -------
    prediction_mat: ndarray of shape (#samples, #trees)
    """
    if hasattr(forest, "estimators_"):
        prediction_mat = np.transpose([tree.predict(X) for tree in forest.estimators_])
    else:
        print("请传入支持estimators_属性的森林")
    return prediction_mat

def get_trees_predict_proba(forest, X):
    """获取森林中每一棵树的预测结果的概率
    Parameters
    ----------
    forest: estimator
    X: array_like
    
    Returns
    -------
    proba_mat: ndarray of shape (#samples, #trees * #classes)
    """
    if hasattr(forest, "estimators_"):
        proba_mat = np.hstack([tree.predict_proba(X) for tree in forest.estimators_])
    else:
        print("请传入支持estimators_属性的森林")
    return proba_mat
    
def predict_proba_mat_parallel(ensemble, X) -> arr:
    """计算并行森林的概率矩阵
    Returns
    -------
    proba_mat: ndarray of shape(#estimators of a forest, #samples, #classes)"""
    proba_mat = []
    for tree in ensemble.estimators_:
        proba_mat.append(tree.predict_proba(X))
    return np.array(proba_mat)

def get_precision_f1(y_true, y_pred):
    """ 获取 每个类的精度 和 整体的f1
    
    Returns
    -------
    record: DataFrame, shape = (#classes + 1, ).
        index = ['precision_class_0', 'precision_class_1', ..., 'f1-score']
    """
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    precision_list = []
    labels = np.unique(y_true)
    for label in labels:
        precision_list.append(report_dict[str(label)]['precision'])
    precision_list.append(report_dict['macro avg']['f1-score'])
    index = [f"precision_class_{label}" for label in labels]
    index.append('f1-score')
    return pd.DataFrame(precision_list, index=index)


