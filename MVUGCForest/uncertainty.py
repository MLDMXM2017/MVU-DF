import numpy as np
from numpy import ndarray as arr
from typing import Union, Tuple, List

from .uncertainty_utils import *
# from uncertainty_utils import *


def get_uncertainty(evidence_mat: arr, method: str="entropy", split: bool=False) -> Union[arr,Tuple[arr, arr]]:
    """计算给定概率矩阵下的不确定度

    Parameters
    ----------
    evidence_mat: ndarray, shape is (#base_estimators, #samples, #classes). 
        概率矩阵或者证据矩阵
    method: str, "entropy"(default) or "evidence"
        The method for calculating entropy.

    Returns
    -------
    uncertainty: ndarray or tuple(ndarray, ndarray), 
        shape is (#samples, ).
    S: ndarray of shape = (#samples, ).
         Dirichlet strength
    """
    if method == "entropy":
        proba_ave = np.mean(evidence_mat, axis=0)
        u_t = calculate_entropy(proba_ave) # total uncertainty

        if split:
            u_a = np.mean([calculate_entropy(proba) for proba in evidence_mat], axis=0)
            u_e = u_t - u_a
            return (u_a, u_e)
        return u_t
    # 基于证据计算不确定度
    elif method == "evidence":
        K = np.shape(evidence_mat)[1]
        S = (np.sum(evidence_mat, axis=1)+K)
        u_t = K/S
        return u_t, S
    # 利用决策树计算不确定度
    elif method == "decision_tree":
         
        pass
    
def evidence_to_opinion(evidence_mat: arr) -> arr:
    """将evidence转化为opinion
    
    Parameters
    ----------
    evidence_mat: arr, shape=(#samples, #classes)
    
    Returns
    -------
    opinion: arr, shape=(#samples, #classes + 1)
    """
    u_t, S = get_uncertainty(evidence_mat, method="evidence")
    # convert evidence_mat to belief mat
    belief_mat = evidence_mat / S[:, np.newaxis]
    opinion = np.hstack([belief_mat, u_t[:, np.newaxis]])
    return opinion

def get_opinion_base_proba_mat(mat: arr, est_type: str) -> arr:
    """基于概率矩阵计算计算opinion
        原来的名字: get_opinion 已经弃用
    
    Parameters
    ----------
    mat: ndarray 
        概率矩阵 或 margin value矩阵
        如果是概率矩阵, shape (#base_estimators, #samples, #classes)
        如果是margin value矩阵: shape (#sample, #classes)
    est_type: str, 
        树的种类
        parallel forest: ["RandomForestClassifier", "ExtraTreesClassifier"]
        boosting forest: ["XGBClassifier"]

    Returns
    -------
    opinion_mat: ndarray, shape=(#samples, #classes + 1) 
    """
    if est_type in ["XGBClassifier", "LGBMClassifier"]:
        margin_value = mat
        if len(np.shape(margin_value)) == 1:
            assert False, "请为XGBoost二分类使用 'multi:softmax' 优化目标函数. "
        elif len(np.shape(margin_value)) == 2:
            evidence_mat = np.where(margin_value<0, 0., margin_value)
            opinion_mat = evidence_to_opinion(evidence_mat)
        
    elif est_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        proba = mat
        # 计算总不确定度
        u_ = get_uncertainty(proba, method="entropy")[:, np.newaxis]
        # 限制最小的不确定度为1-1e-7
        u_ = np.where(u_==1, 1-(1e-7), u_)

        # 计算各个类的置信度
        b_t = 1-u_
        proba_sum = np.sum(proba, axis=0)
        b_mat = (proba_sum / np.sum(proba_sum, axis=1)[:, np.newaxis]) * b_t

        opinion_mat = np.hstack([b_mat, u_])
    return opinion_mat

def get_opinion_base_evidence(evidence:arr, W=2.) -> arr:
    """基于证据计算opinion

    Parameters
    ----------
    evidence_mat: ndarray of shape(#samples, #classes)
        证据矩阵
    W: float | None
        非信息先验权重
    
    Returns
    -------
    opinion: ndarray of shape(#samples, #classes+1)
        有可能会返回空洞的观点!
    """
    if isinstance(W, (np.ndarray, list)):
        W = W.reshape((-1,1))
    else:
        W = np.zeros((len(evidence),1)) + W # shape of (#sample, 1)
        
    opinion = np.hstack([evidence, W])
    opinion = opinion/np.sum(opinion, axis=1,keepdims=True)
    return opinion

def joint_opinions_1(opinion1: arr, opinion2: arr) -> arr:
    """计算两个opinion的联合opinion, 
        refer to 'TRUSTED MULTI-VIEW CLASSIFICATION'
    """
    combining_B = np.array(
        [np.outer(opinion1[i], opinion2[i]) for i in range(opinion1.shape[0])])
    joint_opinion = np.empty_like(opinion1)

    mask = np.ones_like(combining_B[0], dtype=bool) # 创建一个与mat形状相同的掩码数组，并将所有元素设置为True
    mask[:, -1] = False # 将最后一行元素的掩码设置为False
    mask[-1, :] = False # 将最后一列元素的掩码设置为False
    mask[np.diag_indices_from(mask)] = False # 将对角线元素的掩码设置为False
    C = np.array([np.sum(mat[mask])  for mat in combining_B]) # 计算C

    scale_factor = 1 / (1-C)
    for i in range(joint_opinion.shape[1]-1):
        joint_opinion[:,i] = scale_factor*(combining_B[:,i,i]+combining_B[:,-1,i]+combining_B[:,i,-1])
    joint_opinion[:, -1] = scale_factor*combining_B[:, -1, -1]
    return joint_opinion

def joint_opinions_2(opinion1:arr, opinion2:arr):
    """两个opinion的联合opinion
        考虑到了当两个view之间的意见冲突时, 联合不确定度应该增加, 而不是减少
    """
    combining_B = np.array(
        [np.outer(opinion1[i], opinion2[i]).T for i in range(opinion1.shape[0])]) # shape of (#samples, #classes+1, #classes+1)
    joint_opinion = np.empty_like(opinion1)

    mask = np.ones_like(combining_B[0], dtype=bool) # 创建一个与mat形状相同的掩码数组，并将所有元素设置为True
    mask[:, -1] = False # 将最后一行元素的掩码设置为False
    mask[-1, :] = False # 将最后一列元素的掩码设置为False
    mask[np.diag_indices_from(mask)] = False # 将对角线元素的掩码设置为False
    C = np.array([np.sum(mat[mask])  for mat in combining_B]) # 计算C, shape of (#samples, )

    # 预测结果
    y_pred_1 = np.argmax(opinion1[:, :-1], axis=1)
    y_pred_2 = np.argmax(opinion2[:, :-1], axis=1)
    
    u1 = opinion1[:, -1]
    u2 = opinion2[:, -1]
    u_mul = u1*u2
    u_mul_signed = np.where(y_pred_1==y_pred_2, -u_mul, u_mul)
    u_mean = np.mean([u1, u2], axis = 0)

    delta = 1-C-u_mul+u_mean
    
    scale_factor = 1/(delta+u_mul_signed)

    # 计算联合belief, 遍历每个类
    for i in range(joint_opinion.shape[1]-1):
        joint_opinion[:,i] = scale_factor*(combining_B[:,i,i]+combining_B[:,-1,i]+combining_B[:,i,-1])
    # 计算联合uncertainty
    joint_opinion[:, -1] = scale_factor * (u_mul_signed+u_mean) 

    return joint_opinion

def joint_multi_opinion(opinion_list: List[arr], method:int=2)->arr:
    """计算多个opinion的联合opinion"""
    
    assert len(opinion_list)>1, "opinion_list参数应该包含至少两个opinion"
    if method == 1:
        joint_func = joint_opinions_1
    elif method == 2:
        joint_func = joint_opinions_2

    joint_opinion = opinion_list[0]
    for opinion in opinion_list[1:]:
        joint_opinion = joint_func(joint_opinion, opinion)

    return joint_opinion

def opinion_to_proba(opinion: arr)->arr:
    """将opinion转为为概率
    有3种可选方案: 
    1. 去掉uncertainty之后作归一化
    2. 将uncertainty平分到所有类的概率上
    3. 直接使用belief作为probability
    
    Parameters
    ----------
    opinion: ndarray, 有可能会存在空洞的观点!
    """
    method = 1
    if method == 1:
        proba = opinion[:, :-1].copy()
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
    elif method == 2:
        proba = opinion[:, :-1].copy() 
        proba += opinion[:, -1]/(opinion.shape[1]-1)
    elif method == 3:
        proba = opinion[:, :-1].copy()
    else:
        raise ValueError("method must be 1, 2 or 3!")
    if np.sum(np.isnan(proba)) != 0:
        np.nan_to_num(proba, 0)
    return proba

def opinion_to_label(opinion: arr)->arr:
    """将opinion转化为类标签"""
    proba = opinion_to_proba(opinion)
    y_pred_label = np.argmax(proba).reshape(-1)
    return y_pred_label
    

