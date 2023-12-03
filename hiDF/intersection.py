'''
Description: 特征交互
Author: tanqiong
Date: 2023-05-14 20:52:30
LastEditTime: 2023-07-14 11:37:49
LastEditors: tanqiong
'''
import numpy as np
from math import ceil
from .hiDF_utils_threshold import get_rit_tree_data, get_rf_tree_data, _get_stability_score
from .ensemble.wrf import RandomForestClassifierWithWeights, ExtraTreesClassifierWithWeights
# from ..MVUGCForest.layer import Layer
from sklearn.utils import resample
from typing import Dict
import pandas as pd

def feature_intersection(
        layer,
        # layer: Layer, 
        X_train: np.ndarray, y_train: np.ndarray,  
        X_test: np.ndarray, y_test: np.ndarray,    # 测试数据(没用, 占位符)
        use_RIT: bool, 
        propn_n_samples:float,                     # 用于构建RIT时进行boostrap重新采样的比例, 取值区间在[0, 1]
        B: int,                                    # Boostrap的次数(每个RIT都在训练集上进行boostrap采样)
        M: int,                                    # 每次Boostrap后, 构建的RIT的数量, 默认20, 区间[10, 20]或10/20
        new_feature_limit: int,                    # 新特征的数量上限， 默认为6
        n_estimators_bootstrap:int,                # 构建RFW时的数木棵树, 默认为10
        signed: bool,                              # gRIT算法: 是否使用有符号特征交互, 默认为True
        threshold: bool,                           # gRIT算法: 筛选特征时是否使用阈值进行筛选, 默认为True

        max_depth_RIT: int,                        # RIT树的最大深度, 4或5
        random_state:int = None,                   # 
        stability_threshold = 0.5,                 # 稳定性阈值, 0.5
        bin_class_type=None,                       # 
        noisy_split=False,                         # 
        num_splits=2,
        enforce_intra=True,                               # 是否强迫生成intra-section feature
        new_feature_least=2,                         # 强迫生成intra-section的数量
        enforce_method:str="comb_single",           # 强制intra-section的方法, "comb_single"/"random_select", 组合高频的单一特征/随机选择交互规则
        ):
    ## RIT parameters...
    X_test = None
    y_test = None
    use_RIT = True
    propn_n_samples=0.7  # 0.5
    B=10
    M=20 # 10/20
    new_feature_limit = 6
    n_estimators_bootstrap=100  ## old:5 , maybe too small
    signed=True
    threshold=True
    max_depth_RIT=5 # 4/5
    stability_threshold = 0.15
    bin_class_type=None
    noisy_split=False
    num_splits=2

    if random_state is not None:
        random_state += random_state*layer.layer_id

    if use_RIT :

        def interact_to_feature(interact, features_str_lst):

            thresholds_lst = bootstrap_interact_threshold[interact]
            
            if len( thresholds_lst ) != len( features_str_lst ):
                raise ValueError("interaction: ")
            
            new_feature =  []
            
            for idx, feature in enumerate(features_str_lst):
                ## comp: {'feature_id': int, 'L_R': +1/-1 , 'threshold': double }
                new_feature_comp = { 'feature_id': int(feature[:-1]) , 'threshold': thresholds_lst[idx] }  
                if feature[-1] == 'L':
                    new_feature_comp['L_R'] = -1
                else:
                    new_feature_comp['L_R'] = 1

                new_feature.append( new_feature_comp )
            
            return new_feature
        
        ## 所有randomforest feature importances
        all_rf_feature_importances = []
        for node in layer.nodes:
            for rf in node.estimators_.values() :
                all_rf_feature_importances.append( rf.feature_importances_[ :X_train.shape[1] ] )

        n_samples = ceil( propn_n_samples * X_train.shape[0])

        all_K_iter_rf_data = {}

        all_rf_weights = {}

        # Initialize dictionary of bootstrap rf output
        # 初始化随机森林boostrap的字典
        all_rf_bootstrap_output = {}

        # 初始化RIT的boostrap的字典
        # Initialize dictionary of bootstrap RIT output
        all_rit_bootstrap_output = {}


        ##  
        
        bootstrap_feature_importance = np.zeros( (X_train.shape[1] , ) )
        for b in range(B):

            X_train_rsmpl, y_rsmpl = resample( X_train, y_train, replace=False, n_samples=n_samples ) #, stratify = y_train)

            rf_bootstrap = RandomForestClassifierWithWeights(n_estimators=n_estimators_bootstrap, 
                                                             n_jobs=-1,
                                                             max_features=None,  # None/'sqrt'
                                                             max_depth=None, 
                                                             min_samples_leaf=3, 
                                                             random_state=random_state,# 
                                                             )
        
            # Fit RF(w(K)) on the bootstrapped dataset
            rf_bootstrap.fit( X=X_train_rsmpl, y=y_rsmpl , feature_weight= all_rf_feature_importances[ b % len(all_rf_feature_importances)] )

            bootstrap_feature_importance += rf_bootstrap.feature_importances_

            
            all_rf_tree_data = get_rf_tree_data(rf=rf_bootstrap, 
                                                X_train=X_train_rsmpl,  
                                                X_test=X_test, 
                                                y_test=y_test,
                                                signed=signed,  
                                                threshold=threshold 
            )

            # Update the rf bootstrap output dictionary
            all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

            # Run RIT on the interaction rule set
            # CHECK - each of these variables needs to be passed into
            # the main run_rit function
            all_rit_tree_data = get_rit_tree_data(
                all_rf_tree_data=all_rf_tree_data, bin_class_type=bin_class_type,
                M=M, max_depth=max_depth_RIT, noisy_split=noisy_split, num_splits=num_splits)

            # Update the rf bootstrap output dictionary
            # We will reference the RIT for a particular rf bootstrap
            # using the specific bootstrap id - consistent with the
            # rf bootstrap output data
            all_rit_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rit_tree_data

        bootstrap_interact_stability, bootstrap_interact_threshold = _get_stability_score(all_rit_bootstrap_output)

        
        new_features_lst = []
        new_feature_stability =[]
        single_high_freq_lst = []           # 高频的单特征
        single_freq_lst = []

        bootstrap_interact_stability = {k: v for k, v in sorted(bootstrap_interact_stability.items(), key=lambda item: item[1], reverse=True)}
        
        added_new_num = 0
        for interact, stability in bootstrap_interact_stability.items():
            
            # if (added_new_num >= new_feature_limit or stability < stability_threshold):
            #     break
            
            features_str_lst = interact.strip().split('_')

            if len(features_str_lst)>1 and (stability>=stability_threshold) : 
                new_feature = interact_to_feature(interact, features_str_lst)

                added_new_num += 1

                new_features_lst.append( new_feature )
                new_feature_stability.append( stability )
            elif len(features_str_lst)==1:
                single_high_freq_lst.append( interact )
                single_freq_lst.append( stability )
            
            if (added_new_num >= new_feature_limit or stability < stability_threshold) and len(single_high_freq_lst)>=3:
                break

        # 如果没有生成特征交互项
        if (added_new_num < new_feature_least) and enforce_intra:
            if enforce_method=="random_select":
                
                # 随机选择n条交互规则
                interact_list = list( bootstrap_interact_stability.keys() )
                interact_length = np.array( [len(interact.split('_')) for interact in interact_list ] )
                mean_length = np.mean(interact_length)
                idx_interact_select = np.random.choice(
                    np.argwhere( (interact_length>1) & (interact_length<mean_length) ).ravel(), 
                    new_feature_least-added_new_num,
                )
                for idx in idx_interact_select:
                    interact = interact_list[idx]
                    stability = bootstrap_interact_stability[interact]
                    features_str_lst = interact.strip().split('_')
                    new_feature = interact_to_feature(interact, features_str_lst)

                    added_new_num += 1
                    
                    new_features_lst.append( new_feature )
                    new_feature_stability.append( stability )
            elif enforce_method == "comb_single":
                try:
                    ids = np.random.choice(range(len(single_high_freq_lst)), size=2)
                except Exception:
                    print(single_high_freq_lst)
                    # ids = np.array([0, 1])
                interact = ""
                stability = np.mean(np.array(single_freq_lst)[ids])
                for idx in ids:
                    interact += single_high_freq_lst[idx] + "_"
                interact = interact.strip("_")
                bootstrap_interact_threshold[interact] = [ bootstrap_interact_threshold[item] for item in np.array(single_high_freq_lst)[ids] ]  # 将新生成的随即规则注册到bootstrap_interact_threshold中
                bootstrap_interact_stability[interact] = stability
                features_str_lst = interact.strip().split('_')
                new_feature = interact_to_feature(interact, features_str_lst)
                
                added_new_num += 1
                
                new_features_lst.append( new_feature )
                new_feature_stability.append( stability )
                
        new_feature_stability = np.array(new_feature_stability)
        # print(f"  intersection feature number:\t{len(new_features_lst)}")
        return new_features_lst, new_feature_stability, bootstrap_feature_importance
    
def intra_generation(X_train, new_features_lst, new_feature_stability, bootstrap_feature_importance):
    """生成模态内交互特征
    
    Parameters
    ----------
    X_train: ndarray, 训练数据
    new_features_lst: list, shape of (#new features, ). 
        the element of new_features_lst is like 
        [{feature_id:int ,L_R: +1/-1, threshold: double}, ...] .
    new_feature_stablitity: List, 新特征的稳定性得分
    bootstrap_feature_importance: List, shape of (#features of X_train, )

    Returns
    -------
    new_features: ndarray of shape(#samples, #new features)
    """
    ####### augment data  ####### 增强数据, 利用RIT的结果增加新的特征

    # _mean, _min, _max, _std = [], [], [], []
    # for i in range( X_train.shape[1] ):
    #     _mean.append( np.mean(X_train[:,i]) )
    #     _min.append( np.min(X_train[:,i]) )
    #     _max.append( np.max(X_train[:,i]) )
    #     _std.append( np.std(X_train[:,i]) )
    """
    X_train = X_train.astype(float)
    _mean = np.mean(X_train, axis=0).tolist()
    _min = np.min(X_train, axis=0).tolist()
    _max = np.max(X_train, axis=0).tolist()
    _std = np.std(X_train, axis=0).tolist()
    """

    new_train_augment = None
    ## One feature
    pruned_new_feature_stability = []

    for idx, new_feature in enumerate(new_features_lst):       
        
        pruned_new_feature_stability.append( new_feature_stability[idx] )

        train_augment = np.zeros( (X_train.shape[0], ) )

        for idx2, feature_comp in enumerate(new_feature):
            temp = X_train[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
            #temp = ( temp ) / _std[feature_comp['feature_id']]
            #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
            temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
            #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
            train_augment += temp.astype(np.float64)
            

        ### relu
        train_augment = np.maximum( train_augment, 0 )

        train_augment = train_augment.reshape(-1,1)

        ## augment train and test
        if idx == 0:
            new_train_augment = train_augment
        else:
            new_train_augment = np.concatenate( [new_train_augment, train_augment] , axis=1 )
    
    # 如果没有生成交互的增强特征, 则需要对齐训练数据
    if new_train_augment is None:
        new_train_augment = np.empty((X_train.shape[0], 0)) 
    
    return new_train_augment

def inter_operator(intra_dict:Dict[int, np.ndarray], operations:list=None,
                     random_state:int=None):
    """模态之间的特征交互, 使用运算符生成交互特征
    Parameters
    ----------
    x_dict: dict, 
    operating: list, shape of (n_combinations, n_views), 操作符列表, 

    Returns
    -------
    x_new: ndarray of shape(#samples, #new features)
    """

    intra_dict:Dict[int, pd.DataFrame] = {v:pd.DataFrame(intra) for v, intra in intra_dict.items()}
    intra_cols = { v:intra.columns.to_list() for v, intra in intra_dict.items()}
    import itertools
    # 提取每个字典值的一个元素，如果值为空列表，则使用一个占位元素
    intra_cols_list = [v if v else ["placeholder"] for v in intra_cols.values()]
    # 生成所有可能的组合
    combinations = list(itertools.product(*intra_cols_list))
    inter_list = []

    operators = ["add", "sub", "mul"]
    if operations is None:
        # 随机选择操作符
        np.random.seed(random_state)
        operations = np.random.choice( 
            operators, 
            size=( len(combinations), len(combinations[0]) ),
        )
    # 对每一种模态间的特征组合
    for i, comb in enumerate(combinations):
        inter = None
        for v,col in enumerate(comb):
            operator = operations[i][v]

            if col == 'placeholder': continue
            if inter is None:
                inter = intra_dict[v][col]
            else:
                if operator == "add":
                    inter = inter + intra_dict[v][col]
                elif operator == "sub":
                    inter = inter - intra_dict[v][col]
                elif operator == "mul":
                    inter = inter * intra_dict[v][col]
                else:
                    raise ValueError(f"Not define operator: {operator}" )
        if inter is not None:
            inter_list.append(inter)
    inter_list = np.transpose(inter_list)
    if len(inter_list) == 0:
        inter_list = np.empty((intra_dict[0].shape[0], 0))
    return inter_list, operations

def selection_base_fisher(x, y, rate):
    """基于fisher ratio进行特征选择"""
    import pandas as pd
    df = pd.DataFrame(x)
    class_column = 'label'
    df[class_column] = y

    classes = df[class_column].unique()
    if len(classes) != 2:
        raise ValueError("Fisher ratio can only be computed for binary classification problems.")

    class1 = df[df[class_column] == classes[0]].drop(columns=[class_column])
    class2 = df[df[class_column] == classes[1]].drop(columns=[class_column])

    means_diff = np.abs(class1.mean() - class2.mean())
    var_sum = class1.var() + class2.var()

    # compute fisher ratio for each feature and find the one with the maximum value
    fisher_ratios = means_diff / var_sum
    max_fisher_ratio_feature = fisher_ratios.idxmax()

    return max_fisher_ratio_feature, fisher_ratios[max_fisher_ratio_feature]

def select_besk_k(x, y=None, k:int=10, support:list=None):
    """选择最好的k个特征"""
    from sklearn.feature_selection import SelectKBest, f_classif
    if x.shape[1] <= k:
        return x, []
    if (y is not None) and (support is None):
        if (x.shape[1]>=k):
            selector = SelectKBest(t_test, k=k)
            x = selector.fit_transform(x, y)
            support = selector.get_support()
    else:
        x = x[:, support]
    return x, support

def t_test(x, y):
    """t_test

    Parameters
    ----------


    Returns
    -------
    F : array, shape = [n_features,]
    The set of F values.

    pval : array, shape = [n_features,]
    The set of p-values.
    """
    
    from scipy import stats

    t_test_results = []
    for i in range(x.shape[1]):
        cat1 = x[y==0, i]
        cat2 = x[y==1, i]
        t_test_results.append(stats.ttest_ind(cat1, cat2))
    t_test_results = np.array(t_test_results)
    return (t_test_results[:,0], t_test_results[:, 1])

def remove_highly_correlated_features(array, threshold=0.7, remove_idx=None):
    """移除具有高相关性的特征对"""
    if array.shape[1] <= 1:
        return array, []
    # 如果remove_idx是None
    if remove_idx is None:
        # 计算特征之间的相关系数矩阵
        corr_matrix = np.corrcoef(array, rowvar=False)

        # 获取上三角矩阵的索引
        triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)

        # 创建一个集合，用于存储要移除的特征索引
        features_to_remove = set()

        # 根据相关系数大于阈值的特征对，找出要移除的特征索引
        for i, j in zip(triu_indices[0], triu_indices[1]):
            if np.abs(corr_matrix[i, j]) > threshold and (i not in features_to_remove):
                features_to_remove.add(j)
        remove_idx = np.unique(list(features_to_remove)).tolist()
        
    # 移除相关性大于阈值的特征
    array_filtered = np.delete(array, remove_idx, axis=1)

    return array_filtered, remove_idx