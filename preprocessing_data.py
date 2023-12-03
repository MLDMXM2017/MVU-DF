import numpy as np
from typing import Tuple, List, Iterator, Dict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_blobs

def load_simulation_multiview_data(random_state=None) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """模拟简单数据集"""
    n_samples = 200
    n_feature_list = [10, 5, 8]     
    n_class = 2
    n_view = len(n_feature_list)

    feature_dict = {}
    np.random.seed(random_state)
    labels = np.random.randint(n_class, size=n_samples)
    for v, n_features in enumerate(n_feature_list):
        _, sample_counts = np.unique(labels, return_counts=True)
        feature_dict[v], label = make_blobs(n_samples=sample_counts,
                                             n_features=n_features,
                                             shuffle=False,
                                             random_state=random_state,)
    return feature_dict, labels

def split_multiview_data(x_dict: Dict[int, np.ndarray], y, cv=None) -> \
    Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """为multi-view数据划分训练, 测试集"""
    if cv is None:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=666)
        cv = [(t,v) for (t,v) in rskf.split(x_dict[0], y)]
    y = y.squeeze()
    train_idx, test_idx = cv[0]
    x_train_dict = {i:item[train_idx].copy() for i, item in enumerate(x_dict.values())}
    x_test_dict = {i:item[test_idx].copy() for i, item in enumerate(x_dict.values())}
    y_train = y[train_idx]
    y_test = y[test_idx]
    return x_train_dict, x_test_dict, y_train, y_test

def split_multiview_data_cv(x_dict: Dict[int, np.ndarray], y, cv=None, random_state=None) -> \
    Iterator[Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]]:
    """为multi-view数据划分用于交叉验证的训练, 测试集"""
    if cv is None:
        if random_state is None: random_state=666
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)
        cv = [(t,v) for (t,v) in rskf.split(x_dict[0], y)]
    y = y.squeeze()
    for train_idx, test_idx in cv:
        x_train_dict = {i:item[train_idx].copy() for i, item in enumerate(x_dict.values())}
        x_test_dict = {i:item[test_idx].copy() for i, item in enumerate(x_dict.values())}
        y_train = y[train_idx]
        y_test = y[test_idx]
        yield x_train_dict, x_test_dict, y_train, y_test