#!/usr/bin/python

import numpy as np
from sklearn import metrics
from . import tree
from .tree import _tree
from functools import partial
from functools import reduce
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.base import (clone,  ClassifierMixin, RegressorMixin)
from .utils import get_rf_tree_data, get_marginal_importances

# Needed for the scikit-learn wrapper function
from sklearn.utils import resample
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor)
from .ensemble import (wrf, wrf_reg, RandomForestClassifierWithWeights, RandomForestRegressorWithWeights, ExtraTreesClassifierWithWeights)

import datetime
import time

import warnings

try:
    from xgboost import XGBClassifier
except ImportError:
    warnings.warn('No xgboost installed')


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from hiDF.gcForest.layer import *



from math import ceil

# Needed for FPGrowth
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.types import *




# Random Intersection Tree (RIT)

def get_rit_tree_data(all_rf_tree_data,
                      bin_class_type=None,
                      M=10,  # number of trees (RIT) to build
                      max_depth=3,
                      noisy_split=False,
                      num_splits=2):
    
    all_rit_tree_outputs = {}
    
    # Create the weighted randomly sampled paths as a generator 作用：随机获取1个训练样本， 得到它的叶子路径
    gen_random_leaf_paths = generate_rit_samples(
        all_rf_tree_data=all_rf_tree_data,
        bin_class_type=bin_class_type)

    for idx, rit_tree in enumerate(range(M)):
        # Create the RIT object
        rit = build_tree_threshold(feature_paths=gen_random_leaf_paths,
                         max_depth=max_depth,
                         noisy_split=noisy_split,
                         num_splits=num_splits)

        # Get the intersected node values
        # CHECK remove this for the final value
        rit_intersected_values = [
            node[1]._val for node in rit.traverse_depth_first()]
        # Leaf node values i.e. final intersected features
        rit_leaf_node_values = [node[1]._val for node in rit.leaf_nodes()]

        
        rit_output = {"rit": rit,
                      "rit_intersected_values": rit_intersected_values,
                      "rit_leaf_node_values": rit_leaf_node_values,
                      #"rit_leaf_node_union_value": rit_leaf_node_union_value
                    }
        # Append output to our combined random forest outputs dict
        all_rit_tree_outputs["rit{}".format(idx)] = rit_output

    return all_rit_tree_outputs


# FILTERING leaf paths
# Filter Comprehension helper function


def _dtree_filter_comp(dtree_data,
                       filter_key,
                       bin_class_type):
    """
    List comprehension filter helper function to filter
    the data from the `get_tree_data` function output

    Parameters
    ----------
    dtree_data : dictionary
        Summary dictionary output after calling `get_tree_data` on a
        scikit learn decision tree object

    filter_key : str
        The specific variable from the summary dictionary
        i.e. `dtree_data` which we want to filter based on
        leaf class_names

    bin class type : int
        Takes a {0,1} class-value depending on the class
        to be filtered

    Returns
    -------
    tree_data : list
        Return a list containing specific tree metrics
        from the input fitted Classifier object

    """

    # Decision Tree values to filter
    dtree_values = dtree_data[filter_key]

    # Filter based on the specific value of the leaf node classes
    leaf_node_classes = dtree_data['all_leaf_node_classes']

    # perform the filtering and return list
    return [i for i, j in zip(dtree_values,
                              leaf_node_classes)
            if bin_class_type is None or j == bin_class_type]


def filter_leaves_classifier(dtree_data,
                             bin_class_type):
    """
    Filters the leaf node data from a decision tree
    for either {0,1} classes for iRF purposes

    Parameters
    ----------
    dtree_data : dictionary
        Summary dictionary output after calling `get_tree_data` on a
        scikit learn decision tree object

    bin class type : int
        Takes a {0,1} class-value depending on the class
        to be filtered

    Returns
    -------
    all_filtered_outputs : dict
        Return a dictionary containing various lists of
        specific tree metrics for each leaf node from the
        input classifier object
    """

    filter_comp = partial(_dtree_filter_comp,
                          dtree_data=dtree_data,
                          bin_class_type=bin_class_type)

    # Get Filtered values by specified binary class

    # unique feature paths from root to leaf node
    uniq_feature_paths = filter_comp(filter_key='all_uniq_leaf_paths_features')

    # total number of training samples ending up at each node
    tot_leaf_node_values = filter_comp(filter_key='tot_leaf_node_values')

    # depths of each of the leaf nodes
    leaf_nodes_depths = filter_comp(filter_key='leaf_nodes_depths')

    # validation metrics for the tree
    #validation_metrics = dtree_data['validation_metrics']

    # return all filtered outputs as a dictionary
    all_filtered_outputs = {"uniq_feature_paths": uniq_feature_paths,
                            "tot_leaf_node_values": tot_leaf_node_values,
                            "leaf_nodes_depths": leaf_nodes_depths,
                            #"validation_metrics": validation_metrics
                            }

    return all_filtered_outputs


def weighted_random_choice(values, weights):
    """
    Discrete distribution, drawing values with the frequency
    specified in weights.
    Weights do not need to be normalized.
    Parameters:
        values: list of values 
    Return:
        a generator that do weighted sampling
    """
    if not len(weights) == len(values):
        raise ValueError('Equal number of values and weights expected')
    if len(weights) == 0:
        raise ValueError("weights has zero length.")

    weights = np.array(weights)
    # normalize the weights
    weights = weights / weights.sum()
    dist = stats.rv_discrete(values=(range(len(weights)), weights))
    #FIXME this part should be improved by assigning values directly
    #    to the stats.rv_discrete function.  -- Yu

    while True:
        yield values[dist.rvs()]


def generate_rit_samples(all_rf_tree_data, bin_class_type=None , return_path_and_weights = False):
    

    # Number of decision trees
    n_estimators = all_rf_tree_data['get_params']['n_estimators']

    all_weights = []
    all_paths = []
    for dtree in range(n_estimators):
        filtered = filter_leaves_classifier(
            dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
            bin_class_type=bin_class_type)
        
        all_weights.extend(filtered['tot_leaf_node_values']) 
        all_paths.extend(filtered['uniq_feature_paths'])

    # Return the generator of randomly sampled observations by specified weights
    if return_path_and_weights :
        return weighted_random_choice(all_paths, all_weights), (all_paths, all_weights)
    else:
        return weighted_random_choice(all_paths, all_weights)


def select_random_path():
    X = np.random.random(size=(80, 100)) > 0.3
    XX = [np.nonzero(row)[0] for row in X]
    # Create the random array generator
    while True:
        yield XX[np.random.randint(low=0, high=len(XX))]



class RITNode2(object):
    """
    A helper class used to construct the RIT Node
    in the generation of the Random Intersection Tree (RIT)
    """

    def __init__(self, val):
        self._val = val
        self._children = []
        #print('  Init node:', val)
        if( len(self._val) != 0 ):
            self._tuple_size = len(val[0]) if ( type(val[0]) is tuple or type(val[0]) is list ) else 1
        else:
            self._tuple_size = -1

    def is_leaf(self):
        return len(self._children) == 0

    @property
    def children(self):
        return self._children

    def add_child(self, val):
        """添加孩子节点
        Parameters
        ----------
        val: triple of (index, 'L' or 'R', threshold). 
        
        """
        if self._tuple_size == -1:
            warnings.warn( "In RIT building: You are trying to add child to a already empty RITNode.  Do nothing ", RuntimeWarning )
            return -1
        
        if self._tuple_size == 1:
            val_intersect = np.intersect1d(self._val, val)
            
        else:
            
            poor_val = [ str(node[0])+node[1] for node in val ]
            poor_self_val = [ str(node[0])+node[1] for node in self._val ]
            
            _ , val_index , self_val_index = np.intersect1d( poor_val, poor_self_val ,return_indices=True )
            if not ( len(val_index) == len(self_val_index) ):
                raise ValueError( "intersect error, not equal sizes returned\n")

            val_intersect=[]
            
            for i , j in zip( val_index, self_val_index ):
                if not ( val[i][0] == self._val[j][0] and val[i][1] == self._val[j][1] ):
                    raise ValueError("After inersect, feature_id and 'L'/'R' are not equal!  -- {} -- {} -- \n".format(val[i], self._val[j] )  )
                
                if val[i][1] == 'L':
                    if self._tuple_size == 3:
                        val_intersect.append( (val[i][0], val[i][1], max( [val[i][2], self._val[j][2] ] ) ) )
                    else: ## tuple_size == 2
                        val_intersect.append( (val[i][0], val[i][1] ) )
                else:
                    if self._tuple_size == 3:
                        val_intersect.append( (val[i][0], val[i][1], min( [val[i][2], self._val[j][2]] ) ) )
                    else: ## tuple_size == 2
                        val_intersect.append( (val[i][0], val[i][1] ) )
    
        #print('    Add RIT node:  intersection of ' , val , ' & ' , self._val , '  --> ' , val_intersect)
        self._children.append(RITNode2(val_intersect))


    def is_empty(self):
        return len(self._val) == 0

    @property
    def nr_children(self):
        return len(self._children) + \
            sum(child.nr_children for child in self._children)

    def _traverse_depth_first(self, _idx):
        ## : generate a 2-element tuple : (id , RITNode)
        yield _idx[0], self
        for child in self.children:
            _idx[0] += 1
            yield from RITNode2._traverse_depth_first(child, _idx=_idx)


class RITTree2(RITNode2):
    """
    Class for constructing the RIT
    """

    def __len__(self):
        return self.nr_children + 1

    def traverse_depth_first(self):
        yield from RITNode2._traverse_depth_first(self, _idx=[0])

    def leaf_nodes(self):
        for node in self.traverse_depth_first():
            if node[1].is_leaf():
                yield node

                #


def build_tree_threshold(feature_paths, max_depth=3,
               num_splits=5, noisy_split=False,
               _parent=None,
               _depth=0):
    """
        Builds out the random intersection tree based
        on the specified parameters [1]_

        Parameters
        ----------
        feature_paths : generator of list of ints or list of tuples [ (feature_id [, 'L'/'R' [, threshold] ] ) ... ]
        ...

        max_depth : int
            The built tree will never be deeper than `max_depth`.

        num_splits : int
                At each node, the maximum number of children to be added.

        noisy_split: bool
            At each node if True, then number of children to
            split will be (`num_splits`, `num_splits + 1`)
            based on the outcome of a bernoulli(0.5)
            random variable

        References
        ----------
            .. [1] Shah, Rajen Dinesh, and Nicolai Meinshausen.
                    "Random intersection trees." Journal of
                    Machine Learning Research 15.1 (2014): 629-654.
    """

    expand_tree = partial(build_tree_threshold, feature_paths,
                          max_depth=max_depth,
                          num_splits=num_splits,
                          noisy_split=noisy_split)

    if _parent is None:
        tree = RITTree2(next(feature_paths))
        expand_tree(_parent=tree, _depth=0)
        return tree

    else:
        _depth += 1
        if _depth >= max_depth:
            #print('   max depth, return')
            return
        if noisy_split:
            num_splits += np.random.randint(low=0, high=2)
        for i in range(num_splits):
            _flag = _parent.add_child(next(feature_paths))
            if _flag == -1 :
                continue
            added_node = _parent.children[-1]
            if not added_node.is_empty():
                #print('   Added node: ', added_node._val , ' is not empty, grow tree after it' )
                expand_tree(_parent=added_node, _depth=_depth)
            else:
                pass
                #print('   Added node: ', added_node._val , ' is empty' )




# extract interactions from RIT output
def rit_interactions(all_rit_tree_data):
   
    interact_threshold = {}
    
    # loop through all trees
    for k in all_rit_tree_data:
        # loop through all found interactions
        
        #  ( 'rit_intersected_values' / 'rit_leaf_node_values' )
        use_leaf_or_all = 'rit_intersected_values'
        for j in range(len(all_rit_tree_data[k][use_leaf_or_all])):
            # if not empty list:
            if len(all_rit_tree_data[k][use_leaf_or_all][j]) != 0:
                
                # stores interaction as string : eg. np.array([(1, 'L', t_1), (12, 'R', t_12), (23, 'R', t_23)])
                
                # becomes { '1L_12R_23R' : [t_1 , t_12 , t_23] }
                key_iteraction = ''
                key_path = []
                value = []
                temp_L_R = []
                
                for idx, rich_node in enumerate(all_rit_tree_data[k][use_leaf_or_all][j]):
                    key_path.append( str(rich_node[0]) + rich_node[1] ) 
                    value.append( rich_node[2] )
                    temp_L_R.append( rich_node[1] )
                key_iteraction = '_'.join( key_path )

                if( key_iteraction in interact_threshold ):
                    ## key already there, update:
                    for idx, L_R in enumerate( temp_L_R ):
                        if( L_R =='L' ):
                            interact_threshold[key_iteraction][idx] = max( ( interact_threshold[key_iteraction][idx], value[idx] ) )
                        else:
                            interact_threshold[key_iteraction][idx] = min( ( interact_threshold[key_iteraction][idx], value[idx] ) )
                else:
                    interact_threshold[key_iteraction] = value

    return interact_threshold



def _get_histogram(interact_counts, xlabel='interaction',
                   ylabel='stability',
                   sort=False):
    """
    Helper function to plot the histogram from a dictionary of
    count data

    Paremeters
    -------
    interact_counts : dict
        counts of interactions as outputed from the 'rit_interactions' function

    xlabel : str, optional (default = 'interaction')
        label on the x-axis

    ylabel : str, optional (default = 'counts')
        label on the y-axis

    sorted : boolean, optional (default = 'False')
        If True, sort the histogram from interactions with highest frequency
        to interactions with lowest frequency
    """

    if sort:
        data_y = sorted(interact_counts.values(), reverse=True)
        data_x = sorted(interact_counts, key=interact_counts.get,
                        reverse=True)
    else:
        data_x = interact_counts.keys()
        data_y = interact_counts.values()

    plt.figure(figsize=(15, 8))
    plt.clf()
    plt.bar(np.arange(len(data_x)), data_y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(data_x)), data_x, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def _get_stability_score(all_rit_bootstrap_output):
    """
    Get the stabilty score from B bootstrap Random Forest
    Fits with RITs
    
    Return
    ------

    
    """
    
    bootstrap_interact_threshold = {}
    bootstrap_interact_stability = {}

    # Initialize values
    bootstrap_interact = []
    B = len(all_rit_bootstrap_output)

    for b in range(B):
        interact_threshold = rit_interactions(
            all_rit_bootstrap_output['rf_bootstrap{}'.format(b)])
        
        
        interact_temp = list( interact_threshold.keys() )
        bootstrap_interact.append(interact_temp)

        ## union boostrap interaction threshold values
        for interact , threshold in interact_threshold.items():
            if interact in bootstrap_interact_threshold:
                new_threshold = []
                old_threshold = bootstrap_interact_threshold[ interact ]
                if not ( type(threshold) is list and type(old_threshold) is list ):
                    raise TypeError("In _get_stability_score():  threshold " )
                if len(threshold) != len(old_threshold):
                    raise ValueError("In _get_stability_score():  threshold_value and old threshold_value ")

                features_str_lst = interact.strip().split('_')
                for idx, feature_L_R in enumerate(features_str_lst):
                    if feature_L_R[-1] == 'L':
                        new_threshold.append( max(threshold[idx], old_threshold[idx]) )
                    else:
                        new_threshold.append( min(threshold[idx], old_threshold[idx]) )
                
                if len(new_threshold) != len(old_threshold):
                    raise ValueError("In _get_stability_score():  new_threshold and old_threshold")

                bootstrap_interact_threshold[ interact ] = new_threshold

            else:
                bootstrap_interact_threshold[ interact ] = threshold

    ## get all items from list of lists
    def flatten(l): return [item for sublist in l for item in sublist] 
    all_rit_interactions = flatten(bootstrap_interact)
    bootstrap_interact_stability = {m: all_rit_interactions.count(
        m) / B for m in all_rit_interactions}
    
    if len(bootstrap_interact_stability) !=  len( bootstrap_interact_threshold):
        raise ValueError("In _get_stability_score():  bootstrap stability and threshold not equal size! ")
    
    return bootstrap_interact_stability, bootstrap_interact_threshold




def _FP_Growth_get_stability_score(all_FP_Growth_bootstrap_output, bootstrap_num):
    """
    Get the stabilty score from B bootstrap Random Forest
    Fits with FP-Growth
    """

    # Initialize values
    bootstrap_interact = []
    B = len(all_FP_Growth_bootstrap_output)

    for b in range(B):
        itemsets = all_FP_Growth_bootstrap_output['rf_bootstrap{}'.format(b)]
        top_itemsets = itemsets.head(bootstrap_num)
        top_itemsets = list(top_itemsets["items"].map(lambda s: "_".join([str(x) for x in sorted(s)])))
        bootstrap_interact.append(top_itemsets)

    def flatten(l): return [item for sublist in l for item in sublist]
    all_FP_Growth_interactions = flatten(bootstrap_interact)
    stability = {m: all_FP_Growth_interactions.count(
        m) / B for m in all_FP_Growth_interactions}
    return stability



from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score


import time
from hiDF.utils import t_sne_visual, plot3d, plot2d, plot1d

import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))




def run_hiDF(X_train,
            X_test,
            y_train,
            y_test,
            add_gcForest=True,
            rf= None,
            n_estimators = 500,
            rf_bootstrap=None,
            initial_weights = None,
            K=5,
            new_feature_limit = 6,
            B=10,
            n_estimators_bootstrap=10,  ## old:5 , maybe too small
            random_state_classifier=2018,
            signed=True,
            threshold=True,
            propn_n_samples=0.5,  # 0.5
            stability_threshold = 0.5, 
            bin_class_type=None,
            M=20,  # 10/20
            max_depth=4,  # 4/5
            noisy_split=False,
            num_splits=2):  ##  
    """
        Runs the hiDF algorithm.


        Parameters
        ----------
        X_train : array-like or sparse matrix, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        X_test : array-like or sparse matrix, shape = [n_samples, n_features]
            Test vector, where n_samples in the number of samples and
            n_features is the number of features.

        y_train : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values for training.

        y_test : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values for testing.

        rf : RandomForestClassifier/Regressor to fit, it will not be used directly
            Only the parameters of rf will be used.

        rf_bootstrap : RandomForest model to fit to the bootstrap samples, optional
            default None, which means the same as rf

        K : int, optional (default = 7)
            The number of iterations in iRF.

        B : int, optional (default = 10)
            The number of bootstrap samples

        n_estimators : int, optional (default = 20)
            The number of trees in the random forest when computing weights.
        
        signed : bool, optional (default = False)
            Whether use signed interaction or not

        random_state_classifier : int, optional (default = 2018)
            The random seed for reproducibility.

        propn_n_samples : float, optional (default = 0.2)
            The proportion of samples drawn for bootstrap.

        bin_class_type : int, optional (default = 1)
            ...

        max_depth : int, optional (default = 2)
            The built tree will never be deeper than `max_depth`.

        num_splits : int, optional (default = 2)
                At each node, the maximum number of children to be added.

        noisy_split: bool, optional (default = False)
            At each node if True, then number of children to
            split will be (`num_splits`, `num_splits + 1`)
            based on the outcome of a bernoulli(0.5)
            random variable

        n_estimators_bootstrap : int, optional (default = 5)
            The number of trees in the random forest when
            fitting to bootstrap samples

        Returns
        --------
        all_rf_weights: dict
            stores feature weights across all iterations

        all_rf_bootstrap_output: dict
            stores rf information across all bootstrap samples

        all_rit_bootstrap_output: dict
            stores rit information across all bootstrap samples

        stability_score: dict
            stores interactions in as its keys and stabilities scores as the values

    """
    ##  
    all_layers_performance = []

    # Set the random state for reproducibility
    np.random.seed(random_state_classifier)

    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(propn_n_samples * X_train.shape[0])

    # All Random Forest data
    all_K_iter_rf_data = {}

    # Initialize dictionary of rf weights
    # CHECK: change this name to be `all_rf_weights_output`
    all_rf_weights = {}

    # Initialize dictionary of bootstrap rf output
    all_rf_bootstrap_output = {}

    # Initialize dictionary of bootstrap RIT output
    all_rit_bootstrap_output = {}

    t_sne_train = []    

    ### early stopping
    bad = 0

    best_val_acc = 0.0
    best_test_acc=0.0
    best_layer_index = 0

    ### gcForest 
    best_val_acc2 = 0.0
    best_test_acc2=0.0
    best_layer_index2 = 0

    #### augment type
    X_train_eval = X_train.copy()
    X_test_eval = X_test.copy()
    

    temp_y = np.zeros( X_train.shape[1] )
    feature_order = np.ones( X_train.shape[1] )
    feature_depth = np.zeros( X_train.shape[1] )

    for k in range(K):
        
        ### t-SNE
        # t_sne_visual( X_test_eval, y_test , file_name='hiDF', idx=k , random_state=random_state_classifier )   #  X_test_eval, y_test   /  X_train_eval, y_train 
        
        # if k==0:
        #     plot2d( X_test_eval, y_test , file_name='hiDF')
        # else:
        #     plot1d(X_train_eval[:, -1:], y_train)
        

        print( "\n{}\nLayer {} ".format( '-'*40 , k ) )

        ### early stopping
        ##  
        rf_val = RandomForestClassifier(n_estimators=100, n_jobs=-1 ) 
        
        # X1, X_val, y1, y_val = train_test_split( X_train_eval, y_train, random_state=random_state_classifier )
        # rf_val.fit(X1, y1)
        # y_val_pred = rf_val.predict(X_val)
        # new_val_acc = accuracy_score( y_val, y_val_pred )

        cv = StratifiedKFold( 5, shuffle=True, random_state= random_state_classifier )
        val_score = cross_val_score( rf_val, X_train_eval, y_train, scoring='accuracy', cv=cv , n_jobs=-1 )
        new_val_acc = np.mean( val_score )

        print( '\n  cross-validation acc:', new_val_acc )
        

        t_sne_train.append( np.copy(X_train) )
        ## init RF
        rf = RandomForestClassifierWithWeights(n_estimators=n_estimators, n_jobs=-1 )
        
        print( 'Data shape: X_train:{},  X_train_eval:{} , X_test:{}, X_test_eval:{}, Start fitting ...'\
                .format(  X_train.shape, X_train_eval.shape, X_test.shape, X_test_eval.shape ) )
        
        rf.fit( X=X_train_eval, y=y_train )
        
        y_pred = rf.predict( X_test_eval )
        acc = accuracy_score( y_test, y_pred )
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        #print('feature importance: \n{}\n'.format(rf.feature_importances_) )
        print('\nLayer {} fit , test acc:{} ,  micro/macro f1:{}/{} \n'.format( k, acc, micro_f1, macro_f1 ) )
        print('  Forest infomation: avg_max_depth:{}, total_leaf_paths:{}' \
                .format(np.mean([estimator.tree_.max_depth for estimator in rf.estimators_]), rf.n_paths ))
        all_layers_performance.append(acc)


        ###########################################################
        #### marginal importance & entry depth  
        ( marginal_importances , avg_entry_depth  )  = get_marginal_importances( rf, X_train )
        
        ### feature   
        if( len(feature_depth)  <  X_train.shape[1] ):
            feature_depth = np.concatenate( [feature_depth, np.ones(X_train.shape[1]-len(feature_depth))+np.max(feature_depth) ] )

        temp_x = np.concatenate([marginal_importances.reshape(-1,1), avg_entry_depth.reshape(-1,1) ] , axis=1 )
        if( k==0 ):
            tt = 2  
            temp_y[ : int(tt) ] = 1 
        
        else:
            # print( '\n', temp_y.shape, temp_y, '\n', temp_x.shape, temp_x , '\n')
            new_temp_y = np.ones( temp_x.shape[0] - len(temp_y) )
            print( new_temp_y.shape, new_temp_y )
            temp_y = np.concatenate( [temp_y,  np.max(temp_y) + new_temp_y ] )

        # plt.scatter( feature_order , marginal_importances.reshape(-1,1) )    
        # plt.savefig( './fig/{}.png'.format( time.time()) )
        plot2d( np.concatenate([feature_order.reshape(-1,1) , marginal_importances.reshape(-1,1)], axis=1 ), y= feature_depth )
        # input('xxxxxxx')    




        if k>0 and new_val_acc > best_val_acc:
            bad = 0
            best_layer_index = k
            best_val_acc = new_val_acc
            best_test_acc = acc 

        elif new_val_acc <= best_val_acc:
            bad += 1

        if bad >=2 :
            pass
            # break 

        ############################################
        ##### gcForest augment train and test ######
        ############################################
         
        # if add_gcForest:
        #     pass
            
            # new_val_acc2 = compute_accuracy(y_train, val_prob)
            # new_test_acc2 = compute_accuracy( y_test, test_prob )
            
            # print('   Layer {}, gcForest, val acc:{}; test acc:{}'.format(k, new_val_acc2,  new_test_acc2) )

            # if k>0 and new_val_acc2 >= best_val_acc2:
            #     best_layer_index2 = k
            #     best_val_acc2 = new_val_acc2
            #     best_test_acc2 = new_test_acc2


        if k== K-1 :
            break

         
        test_xgboost = False
        if test_xgboost:
            if( len(np.unique(y_train))==2 ):
                xgb_objective = 'binary:logistic'
            else:
                xgb_objective = 'multi:softmax'
            print('\n  xgboost objective:{}'.format(xgb_objective) )
            param_dist = {'objective': xgb_objective, 'n_estimators':100, 'n_jobs':-1 , 'random_state':random_state_classifier }
            clf = XGBClassifier(**param_dist)
            clf.fit(X_train, y_train)
            y_pred = clf.predict( X_test )
            acc = accuracy_score(y_test, y_pred) 
            print('  Layer {} xgboost acc: {}\n'.format( k, acc ) )

         
        test_adaboost = False
        if test_adaboost:
            clf = AdaBoostClassifier( DecisionTreeClassifier(), n_estimators=100, random_state=random_state_classifier )
            clf.fit(X_train, y_train)
            y_pred = clf.predict( X_test )
            acc = accuracy_score(y_test, y_pred) 
            print('  Layer {} AdaBoost acc: {}\n'.format( k, acc ) )


        #permutation_importance_mean = permutation_importance(rf, X_test, y_test, n_repeats=10, n_jobs=-1 )['importances_mean']
        print('  Start bootstrap, RIT...')
        bootstrap_feature_importance = np.zeros( (X_train.shape[1] , ) )
        # Run the RITs
        for b in range(B):

            # Take a bootstrap sample from the training data
            # based on the specified user proportion
            if isinstance(rf, ClassifierMixin):
                X_train_rsmpl, y_rsmpl = resample(
                    X_train, y_train, replace=False, n_samples=n_samples ) #, stratify = y_train)
            else:
                X_train_rsmpl, y_rsmpl = resample(
                    X_train, y_train, n_samples=n_samples)
                
            

            # Set up the weighted random forest
            # Using the weight from the (K-1)th iteration i.e. RF(w(K))
             
            rf_bootstrap = RandomForestClassifierWithWeights(n_estimators=n_estimators_bootstrap, n_jobs=-1 )
            
            # Fit RF(w(K)) on the bootstrapped dataset
            rf_bootstrap.fit( X=X_train_rsmpl, y=y_rsmpl , feature_weight= rf.feature_importances_[:X_train_rsmpl.shape[1]] + np.random.normal(0.0, np.min(rf.feature_importances_), X_train_rsmpl.shape[1] ) )

            bootstrap_feature_importance += rf_bootstrap.feature_importances_

            # All RF tree data
            # CHECK: why do we need y_train here?
            all_rf_tree_data = get_rf_tree_data(
                rf=rf_bootstrap,
                X_train=X_train_rsmpl,
                X_test=X_test,
                y_test=y_test,
                signed=signed,
                threshold= threshold )

            # Update the rf bootstrap output dictionary
            all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

            # Run RIT on the interaction rule set
            # CHECK - each of these variables needs to be passed into
            # the main run_rit function
            all_rit_tree_data = get_rit_tree_data(
                all_rf_tree_data=all_rf_tree_data,
                bin_class_type=bin_class_type,
                M=M,
                max_depth=max_depth,
                noisy_split=noisy_split,
                num_splits=num_splits)

            # Update the rf bootstrap output dictionary
            # We will reference the RIT for a particular rf bootstrap
            # using the specific bootstrap id - consistent with the
            # rf bootstrap output data
            all_rit_bootstrap_output['rf_bootstrap{}'.format(
                b)] = all_rit_tree_data
        #bootstrap_feature_importance /= B

        bootstrap_interact_stability, bootstrap_interact_threshold = _get_stability_score(
            all_rit_bootstrap_output=all_rit_bootstrap_output)

         
        
        new_features_lst = []
        new_feature_stability =[]

        ##  
        bootstrap_interact_stability = {k: v for k, v in sorted(bootstrap_interact_stability.items(), key=lambda item: item[1], reverse=True)}
        # 
        added_new_num = 0
        for interact, stability in bootstrap_interact_stability.items():
            
            if added_new_num >= new_feature_limit or stability < stability_threshold:
                break
            
            features_str_lst = interact.strip().split('_')

            ## this interaction is prevalent, extract features
            if len(features_str_lst)>1 : 
                added_new_num += 1
                thresholds_lst = bootstrap_interact_threshold[interact]
                
                if len( thresholds_lst ) != len( features_str_lst ):
                    raise ValueError("interaction: feature ")
                
                new_feature =  []
                
                for idx, feature in enumerate(features_str_lst):
                    ## comp: {'feature_id': int, 'L_R': +1/-1 , 'threshold': double }
                    new_feature_comp = { 'feature_id': int(feature[:-1]) , 'threshold': thresholds_lst[idx] }  
                    if feature[-1] == 'L':
                        new_feature_comp['L_R'] = -1
                    else:
                        new_feature_comp['L_R'] = 1

                    new_feature.append( new_feature_comp )
                    
                new_features_lst.append( new_feature )
                new_feature_stability.append( stability )

        new_feature_stability = np.array(new_feature_stability)

        
        for item in new_features_lst:
            print( item ) 


        ####### augment data  #######
        
        _mean, _min, _max, _std = [], [], [], []
        for i in range( X_train.shape[1] ):
            _mean.append( np.mean(X_train[:,i]) )
            _min.append( np.min(X_train[:,i]) )
            _max.append( np.max(X_train[:,i]) )
            _std.append( np.std(X_train[:,i]) )

        ## One feature
        pruned_new_feature_stability = []
        new_train_augment = None
        new_test_augment = None
        for idx, new_feature in enumerate(new_features_lst):       
            
            pruned_new_feature_stability.append( new_feature_stability[idx] )

            train_augment = np.zeros( (X_train.shape[0], ) )
            test_augment = np.zeros( (X_test.shape[0], ) )


            
            one_feature_order = 0
            for idx2, feature_comp in enumerate(new_feature):
                temp = X_train[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                #temp = ( temp ) / _std[feature_comp['feature_id']]
                #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
                temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
                train_augment += temp
                
                temp = X_test[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                #temp = ( temp ) / _std[feature_comp['feature_id']]
                #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
                temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
                test_augment += temp

                
                one_feature_order +=  feature_order[ feature_comp['feature_id'] ]

            feature_order = np.append( feature_order, one_feature_order ) 


            ### relu
            train_augment = np.maximum( train_augment, 0 )
            test_augment  = np.maximum( test_augment, 0)

            train_augment = train_augment.reshape(-1,1)
            test_augment = test_augment.reshape(-1,1)

            ## augment train and test
            if idx == 0:
                new_train_augment = train_augment
                new_test_augment = test_augment
            else:
                new_train_augment = np.concatenate( [new_train_augment, train_augment] , axis=1 )
                new_test_augment = np.concatenate( [new_test_augment, test_augment], axis=1 )
        

        

        ## augment train and test
        if ( new_train_augment is not None):
            X_train = np.concatenate( [X_train, new_train_augment] , axis=1 )
            X_test = np.concatenate( [X_test, new_test_augment], axis=1 )


        if ( new_train_augment is not None):
            ## t_sne on  : new only / new+old 
            # t_sne_visual(X_train, y_train, file_name='hiDF', idx=k )
            pass
    
        
        X_train_eval = X_train
        X_test_eval = X_test


        if add_gcForest:
            print('   Add gcForest new feature，  label unique:', len(np.unique(y_train)) )
            kf = StratifiedKFold( 5 , shuffle=True , random_state= random_state_classifier )
            layer = KfoldWarpper(num_forests=5, n_estimators=100, num_classes=len(np.unique(y_train)), n_fold=5, kf=kf, layer_index=k, max_depth=None, min_samples_leaf=1, \
                        sample_weight=None, random_state=random_state_classifier, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 )

            val_prob, val_stack= layer.train(X_train, y_train)
            test_prob, test_stack = layer.predict( X_test )
            
            X_train_eval = np.concatenate([X_train_eval, val_stack], axis=1)
            X_test_eval = np.concatenate([X_test_eval, test_stack], axis=1 )
     

        
    print( '\nbest layer index: {}, its\' test acc: {} \n'.format(best_layer_index, best_test_acc) )
    #print( 'gcForest: best layer index: {}, its\' test acc: {} \n'.format(best_layer_index2, best_test_acc2)  )

        
        


    # print('-'*50, '\n Data shape: X_train:{} , X_test:{}, y_train:{}, y_test:{} , Start fitting ... \n'.format( X_train.shape, X_test.shape, y_train.shape, y_test.shape))


    # #new_feature_weight = np.append( np.ones(X_train.shape[1]-new_train_augment.shape[1]) , new_feature_stability )
    # new_feature_weight = None
    # print('  New feature weight: ', new_feature_weight ,'\n')

    # rf = RandomForestClassifierWithWeights(n_estimators=n_estimators, n_jobs=-1 )
    # rf.fit( X=X_train, y=y_train , feature_weight = new_feature_weight  )
    
    # #print('feature importance: \n{}\n'.format(rf.feature_importances_) )
    # y_pred = rf.predict( X_test )
    # acc = accuracy_score( y_test, y_pred )
    # print('{}\nLast fit , acc:{} \n'.format('-'*40 , acc ) )
    # all_layers_performance.append(acc)


    return all_rf_weights,\
        all_K_iter_rf_data, all_rf_bootstrap_output,\
        all_rit_bootstrap_output, bootstrap_interact_stability, t_sne_train, all_layers_performance, best_test_acc






class gcForest_hi:


    def __init__(self, num_estimator, num_forests, num_classes, max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 , use_metric = 'acc', 
        ## hiDF parameters...
        use_RIT = False,
        new_feature_limit = 6,
        B=10,
        n_estimators_bootstrap=10,  ## old:5 , maybe too small
        signed=True,
        threshold=True,
        propn_n_samples=0.5,  # 0.5
        stability_threshold = 0.5, 
        bin_class_type=None,
        M=20,  # 10/20
        max_depth_RIT=4,  # 4/5
        noisy_split=False,
        num_splits=2
        ):
        
        
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.num_classes = num_classes
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state
        self.purity_function = purity_function
        self.bootstrap = bootstrap
        self.parallel = parallel
        self.num_threads = num_threads
        self.use_metric = use_metric

        self.model = []

        ## RIT 
        self.use_RIT = use_RIT
        self.new_feature_limit = new_feature_limit
        self.B=B
        self.n_estimators_bootstrap=n_estimators_bootstrap
        self.signed=signed
        self.threshold=threshold
        self.propn_n_samples=propn_n_samples        # 用于构建RIT时进行重新采样的比例
        self.stability_threshold =stability_threshold
        self.bin_class_type=bin_class_type
        self.M=M
        self.max_depth_RIT=max_depth_RIT
        self.noisy_split=noisy_split
        self.num_splits=num_splits



    def train(self, X_train, y_train, X_test, y_test, return_test_proba=False ):
        """
        Parameters
        ----------
        return_test_proba"""
         # basis information of dataset
        num_classes = int(np.max(y_train) + 1)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        num_samples, num_features = X_train.shape

        # basis process
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()

        # return value
        val_p = []
        val_acc = []
        best_train_acc = 0.0
        # best_train_acc2 = 0.0
        best_test_acc=0.0
        # best_test_acc2 = 0.0
        layer_index = 0
        best_layer_index = 0
        # best_layer_index2 = 0
        bad = 0
        

        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state)  ##  KFold / StratifiedKFold
        
        np.random.seed(self.random_state)


        while layer_index < self.max_layer:
            
            # t_sne_visual(X_test_new, y_test, file_name='gcForest_HI', idx=layer_index )
            
            print("\n{}\n--------------\nlayer {},   X_train shape:{}...\n ".format(datetime.datetime.now(), str(layer_index), X_train_new.shape ) )

            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                layer_index, self.max_depth, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                self.purity_function, self.bootstrap,  self.parallel, self.num_threads  )

            val_prob, val_stack= layer.train(X_train_new, y_train)
            test_prob, test_stack = layer.predict( X_test_new )

            if self.use_metric == 'acc':
                temp_val_acc = compute_accuracy( y_train, val_prob )
                temp_test_acc = compute_accuracy( y_test, test_prob )
            elif self.use_metric == 'roc_auc':
                if self.num_classes == 2:
                    val_prob = val_prob[:, 1]
                    test_prob = test_prob[:, 1]
                temp_val_acc = roc_auc_score( y_train, val_prob, average="weighted", multi_class="ovr" )
                temp_test_acc = roc_auc_score( y_test, test_prob, average="weighted", multi_class="ovr" )
            elif self.use_metric =='f1' :
                temp_val_acc = f1_score( y_train, np.argmax(val_prob, axis=1), average="weighted")
                temp_test_acc = f1_score( y_test, np.argmax(test_prob, axis=1), average="weighted")
            else:
                raise ValueError('Metric ERROR! ')

            temp_rf_test_acc=0.0
            # rf = RandomForestClassifier(n_estimators=self.num_forests*self.num_estimator, n_jobs=-1 , random_state=self.random_state )
            # rf.fit( X=X_train_new, y=y_train )
            # y_pred = rf.predict( X_test_new )
            # temp_rf_test_acc = accuracy_score( y_test, y_pred )
        
            
            
            if best_train_acc >= temp_val_acc:
                bad += 1
            else:
                bad = 0
                best_train_acc = temp_val_acc
                best_test_acc = temp_test_acc
                best_layer_index = layer_index
                best_test_proba = test_prob
            if bad >= 2:
                # pass
                break

            if( self.max_layer-1 == layer_index ):
                break


            ##### RIT 对拟合后的一层应用RIT算法

            if self.use_RIT :
                print(datetime.datetime.now(), '  ')
                ## 所有randomforest feature importances
                all_rf_feature_importances = []
                for fold in layer.model:
                    for rf in fold.model :
                        all_rf_feature_importances.append( rf.feature_importances_[ :X_train.shape[1] ] )
                print( '    RF feature importances.  '.format(len(all_rf_feature_importances)) )


                n_samples = ceil( self.propn_n_samples * X_train.shape[0])

                all_K_iter_rf_data = {}

                all_rf_weights = {}

                # Initialize dictionary of bootstrap rf output
                # 初始化随机森林boostrap的字典
                all_rf_bootstrap_output = {}

                # 初始化RIT的boostrap的字典
                # Initialize dictionary of bootstrap RIT output
                all_rit_bootstrap_output = {}


                ##  
                print('  Start bootstrap, RIT...')
                bootstrap_feature_importance = np.zeros( (X_train.shape[1] , ) )
                for b in range(self.B):

                    X_train_rsmpl, y_rsmpl = resample( X_train, y_train, replace=False, n_samples=n_samples ) #, stratify = y_train)

                    rf_bootstrap = RandomForestClassifierWithWeights(n_estimators=self.n_estimators_bootstrap, n_jobs=-1 )
                
                    # Fit RF(w(K)) on the bootstrapped dataset
                    rf_bootstrap.fit( X=X_train_rsmpl, y=y_rsmpl , feature_weight= all_rf_feature_importances[ b % len(all_rf_feature_importances)] )

                    bootstrap_feature_importance += rf_bootstrap.feature_importances_

                    
                    all_rf_tree_data = get_rf_tree_data(  rf=rf_bootstrap, X_train=X_train_rsmpl,  X_test=X_test, y_test=y_test,
                        signed=self.signed,  threshold= self.threshold )

                    # Update the rf bootstrap output dictionary
                    all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

                    # Run RIT on the interaction rule set
                    # CHECK - each of these variables needs to be passed into
                    # the main run_rit function
                    all_rit_tree_data = get_rit_tree_data(
                        all_rf_tree_data=all_rf_tree_data, bin_class_type=self.bin_class_type,
                        M=self.M, max_depth=self.max_depth_RIT, noisy_split=self.noisy_split, num_splits=self.num_splits)

                    # Update the rf bootstrap output dictionary
                    # We will reference the RIT for a particular rf bootstrap
                    # using the specific bootstrap id - consistent with the
                    # rf bootstrap output data
                    all_rit_bootstrap_output['rf_bootstrap{}'.format(
                        b)] = all_rit_tree_data


                bootstrap_interact_stability, bootstrap_interact_threshold = _get_stability_score(all_rit_bootstrap_output)

                
                new_features_lst = []
                new_feature_stability =[]

                bootstrap_interact_stability = {k: v for k, v in sorted(bootstrap_interact_stability.items(), key=lambda item: item[1], reverse=True)}
                
                added_new_num = 0
                for interact, stability in bootstrap_interact_stability.items():
                    
                    if added_new_num >= self.new_feature_limit or stability < self.stability_threshold:
                        break
                    
                    features_str_lst = interact.strip().split('_')

                    if len(features_str_lst)>1 : 
                        added_new_num += 1
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
                            
                        new_features_lst.append( new_feature )
                        new_feature_stability.append( stability )

                new_feature_stability = np.array(new_feature_stability)

                
                print( datetime.datetime.now(), '\n  {} , New feature composition:'.format(len(new_features_lst)) )
                # for item in new_features_lst:
                #     print( item ) 


                ####### augment data  ####### 增强数据, 利用RIT的结果增加新的特征
                
                _mean, _min, _max, _std = [], [], [], []
                for i in range( X_train.shape[1] ):
                    _mean.append( np.mean(X_train[:,i]) )
                    _min.append( np.min(X_train[:,i]) )
                    _max.append( np.max(X_train[:,i]) )
                    _std.append( np.std(X_train[:,i]) )

                ## One feature
                pruned_new_feature_stability = []
                new_train_augment = None
                # new_test_augment = None
                for idx, new_feature in enumerate(new_features_lst):       
                    
                    pruned_new_feature_stability.append( new_feature_stability[idx] )

                    train_augment = np.zeros( (X_train.shape[0], ) )
                    test_augment = np.zeros( (X_test.shape[0], ) )


                    

                    for idx2, feature_comp in enumerate(new_feature):
                        temp = X_train[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                        #temp = ( temp ) / _std[feature_comp['feature_id']]
                        #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
                        temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                        #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
                        train_augment += temp.astype(np.float64)
                        
                        temp = X_test[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                        #temp = ( temp ) / _std[feature_comp['feature_id']]
                        #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
                        temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                        #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
                        test_augment += temp.astype(np.float64)

                    ### relu
                    train_augment = np.maximum( train_augment, 0 )
                    test_augment  = np.maximum( test_augment, 0)

                    train_augment = train_augment.reshape(-1,1)
                    test_augment = test_augment.reshape(-1,1)

                    ## augment train and test
                    if idx == 0:
                        new_train_augment = train_augment
                        new_test_augment = test_augment
                    else:
                        new_train_augment = np.concatenate( [new_train_augment, train_augment] , axis=1 )
                        new_test_augment = np.concatenate( [new_test_augment, test_augment], axis=1 )
                

                
                ## augment train and test
                if ( new_train_augment is not None):
                    X_train = np.concatenate( [X_train, new_train_augment] , axis=1 )
                    X_test = np.concatenate( [X_test, new_test_augment], axis=1 )


                layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                    layer_index, self.max_depth, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                    self.purity_function, self.bootstrap,  self.parallel, self.num_threads  )

                val_prob, val_stack= layer.train(X_train, y_train)
                test_prob, test_stack = layer.predict( X_test )
                
                # temp_val_acc2 = compute_accuracy( y_train, val_prob )
                # print('val acc2:', temp_val_acc2)
                # if temp_val_acc2 > best_train_acc2:
                #     best_train_acc2 = temp_val_acc2
                #     best_test_acc2 = temp_test_acc
                #     best_layer_index2 = layer_index
            ##### END RIT


            X_train_new = np.concatenate([X_train, val_stack], axis=1)
            X_test_new = np.concatenate([X_test, test_stack], axis=1 )

        

            layer_index = layer_index + 1

            
        print( 'Total {} layers'.format(len(self.model)) )

        print( '\nBest layer index: {}'.format(best_layer_index ) )
        if return_test_proba:
            return best_test_acc, best_test_proba
        return best_test_acc



    def predict_proba(self, X_test):
        X_test_new = X_test.copy()
        test_prob = []
        for layer in self.model:
            test_prob, test_stack = layer.predict(X_test_new)
            X_test_new = np.concatenate([X_test, test_stack], axis=1)
            layer_index = layer_index + 1

        return test_prob

    def predict(self, X_test):
        X_test_new = X_test.copy()
        test_prob = []
        for layer in self.model:
            test_prob, test_stack = layer.predict(X_test_new)
            X_test_new = np.concatenate([X_test, test_stack], axis=1)

        return np.argmax(test_prob, axis=1)








class gcForest_hi_new:
    '''
    gcForest_HI 
    '''

    def __init__(self, num_estimator, num_forests, num_classes, max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ,
        # hiDF parameters...
        use_RIT = True,
        new_feature_limit = 6,
        B=10,
        n_estimators_bootstrap=10,  ## old:5 , maybe too small
        signed=True,
        threshold=True,
        propn_n_samples=0.5,  # 0.5
        stability_threshold = 0.5, 
        bin_class_type=None,
        M=20,  # 10/20
        max_depth_RIT=4,  # 4/5
        noisy_split=False,
        num_splits=2
        ):
        
        
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.num_classes = num_classes
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state
        self.purity_function = purity_function
        self.bootstrap = bootstrap
        self.parallel = parallel
        self.num_threads = num_threads
        
        self.predictive_model = None
        self.augment_model = None
        self.augment_model_temp = None


        ## RIT 
        self.use_RIT = use_RIT
        self.new_feature_limit = new_feature_limit
        self.B=B
        self.n_estimators_bootstrap=n_estimators_bootstrap
        self.signed=signed
        self.threshold=threshold
        self.propn_n_samples=propn_n_samples  
        self.stability_threshold =stability_threshold
        self.bin_class_type=bin_class_type
        self.M=M
        self.max_depth_RIT=max_depth_RIT
        self.noisy_split=noisy_split
        self.num_splits=num_splits

        self.all_layer_features = []
        self.all_layer_feature_importance = []


    def train(self, X_train, y_train, X_test=None, y_test=None ):

        ### init
        self.predictive_model = None
        self.augment_model = None
        self.augment_model_temp = None
        self.all_layer_features = []
        self.all_layer_feature_importance = []


        # basis information of dataset
        num_classes = int(np.max(y_train) + 1)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        num_samples, num_features = X_train.shape

        # basis process
        X_train_new = X_train.copy()
        
        # return value
        val_p = []
        val_acc = []
        self.best_train_acc = 0.0
        # best_train_acc2 = 0.0
        # best_test_acc2 = 0.0
        layer_index = 0
        self.best_layer_index = 0
        # best_layer_index2 = 0
        bad = 0
        

        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state)  ##  KFold / StratifiedKFold
        
        np.random.seed(self.random_state)


        while layer_index < self.max_layer:
            
            # t_sne_visual(X_train_new, y_train, file_name='gcForest_HI', idx=layer_index )
            
            print("\n--------------\nlayer {},   X_train shape:{}...\n ".format(str(layer_index), X_train_new.shape ) )

            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                layer_index, self.max_depth, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                self.purity_function, self.bootstrap,  self.parallel, self.num_threads  )

            val_prob, val_stack= layer.train(X_train_new, y_train)
            
            temp_val_acc = compute_accuracy( y_train, val_prob )
                        
            # temp_rf_test_acc=0.0
            # rf = RandomForestClassifier(n_estimators=self.num_forests*self.num_estimator, n_jobs=-1 , random_state=self.random_state )
            # rf.fit( X=X_train_new, y=y_train )
            # y_pred = rf.predict( X_test_new )
            # temp_rf_test_acc = accuracy_score( y_test, y_pred )
        
            print( "val  acc:{} ".format( str(temp_val_acc) ) )
            
            
            if self.best_train_acc >= temp_val_acc:
                bad += 1
            else:
                bad = 0

                ### update 2 models
                self.predictive_model = layer
                self.augment_model = self.augment_model_temp
                
                self.best_train_acc = temp_val_acc
                self.best_layer_index = layer_index
                
            if bad >= 2:
                pass
                # break

            if( self.max_layer-1 == layer_index ):
                
                break


            ##### RIT 

            if self.use_RIT :
                print(' {}'.format(datetime.datetime.now()) )
                ## 所有randomforest feature importances
                all_rf_feature_importances = []
                for fold in layer.model:
                    for rf in fold.model :
                        all_rf_feature_importances.append( rf.feature_importances_[ :X_train.shape[1] ] )
                

                n_samples = ceil( self.propn_n_samples * X_train.shape[0])

                all_K_iter_rf_data = {}

                all_rf_weights = {}

                # Initialize dictionary of bootstrap rf output
                all_rf_bootstrap_output = {}

                # Initialize dictionary of bootstrap RIT output
                all_rit_bootstrap_output = {}


                ## 
                print( datetime.datetime.now(), ' Start bootstrap, RIT...')
                bootstrap_feature_importance = np.zeros( (X_train.shape[1] , ) )
                

                parallel_RIT = False

                if not parallel_RIT:
                    for b in range(self.B):
                        
                        X_train_rsmpl, y_rsmpl = resample( X_train, y_train, replace=False, n_samples=n_samples ) #, stratify = y_train)
                        
                        rf_bootstrap = RandomForestClassifierWithWeights(n_estimators=self.n_estimators_bootstrap, n_jobs=-1 )

                        # Fit RF(w(K)) on the bootstrapped dataset
                        ### 0.5s (adult)
                        rf_bootstrap.fit( X=X_train_rsmpl, y=y_rsmpl , feature_weight= all_rf_feature_importances[ b % len(all_rf_feature_importances)] )

                        bootstrap_feature_importance += rf_bootstrap.feature_importances_

                        ### !!!!  1.2s  !!!!(adult)
                        all_rf_tree_data = get_rf_tree_data(  rf=rf_bootstrap, X_train=X_train_rsmpl,  X_test=None, y_test=None,
                            signed=self.signed,  threshold= self.threshold )

                        # Update the rf bootstrap output dictionary
                        all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data


                        # Run RIT on the interaction rule set
                        # CHECK - each of these variables needs to be passed into
                        # the main run_rit function
                        all_rit_tree_data = get_rit_tree_data(
                            all_rf_tree_data=all_rf_tree_data, bin_class_type=self.bin_class_type,
                            M=self.M, max_depth=self.max_depth_RIT, noisy_split=self.noisy_split, num_splits=self.num_splits)

                        # Update the rf bootstrap output dictionary
                        # We will reference the RIT for a particular rf bootstrap
                        # using the specific bootstrap id - consistent with the
                        # rf bootstrap output data
                        all_rit_bootstrap_output['rf_bootstrap{}'.format(
                            b)] = all_rit_tree_data

                
                else:
                    print(' ')
                    pass
                    rit_bootstrap_util_partial = partial( rit_bootstrap_util, X_train=X_train, y_train=y_train, n_samples=n_samples, n_estimators_bootstrap=self.n_estimators_bootstrap, all_rf_feature_importances=all_rf_feature_importances)
                    
                    
                    pool = Pool(10)
                    result = pool.map( rit_bootstrap_util_partial, range(self.B) ) 
                    #close the pool and wait for the work to finish 
                    pool.close() 
                    pool.join() 

                    print(datetime.datetime.now(), ' ')
                    
                    for b, res in enumerate(result):
                        
                        bootstrap_feature_importance += res[0]

                        # Update the rf bootstrap output dictionary
                        all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = res[1]
                        
                        # Update the rf bootstrap output dictionary ; We will reference the RIT for a particular rf bootstrap
                        # using the specific bootstrap id - consistent with the rf bootstrap output data
                        all_rit_bootstrap_output['rf_bootstrap{}'.format(b)] = res[2]
                    
                print(datetime.datetime.now(), ' interaction stability score')

                bootstrap_interact_stability, bootstrap_interact_threshold = _get_stability_score(all_rit_bootstrap_output)

                
                new_features_lst = []
                new_feature_stability =[]

                bootstrap_interact_stability = {k: v for k, v in sorted(bootstrap_interact_stability.items(), key=lambda item: item[1], reverse=True)}
                
                added_new_num = 0
                for interact, stability in bootstrap_interact_stability.items():
                    
                    if added_new_num >= self.new_feature_limit or stability < self.stability_threshold:
                        break
                    
                    features_str_lst = interact.strip().split('_')

                    if len(features_str_lst)>1 : 
                        added_new_num += 1
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
                            
                        new_features_lst.append( new_feature )
                        new_feature_stability.append( stability )

                new_feature_stability = np.array(new_feature_stability)

                
                
                self.all_layer_features.append( new_features_lst )
                self.all_layer_feature_importance.append( bootstrap_feature_importance )


                ####### augment data  #######
                
                ## One feature
                new_train_augment = None
                for idx, new_feature in enumerate(new_features_lst):       
                    
                    train_augment = np.zeros( (X_train.shape[0], ) )
                    

                    for idx2, feature_comp in enumerate(new_feature):
                        temp = X_train[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                        #temp = ( temp ) / _std[feature_comp['feature_id']]
                        #temp = (temp ) / (_max[feature_comp['feature_id']]-_min[feature_comp['feature_id']])
                        temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                        #temp = temp * ( logistic_weight[idx2] * feature_comp['L_R'] )
                        train_augment += temp
                        

                    ### relu
                    train_augment = np.maximum( train_augment, 0 )
                    
                    train_augment = train_augment.reshape(-1,1)
                    
                    ## augment train
                    if idx == 0:
                        new_train_augment = train_augment
                    else:
                        new_train_augment = np.concatenate( [new_train_augment, train_augment] , axis=1 )
                        

                
                ## augment train
                if ( new_train_augment is not None):
                    X_train = np.concatenate( [X_train, new_train_augment] , axis=1 )
                    

                print( datetime.datetime.now(), '  ')
                layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                    layer_index, self.max_depth, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                    self.purity_function, self.bootstrap,  self.parallel, self.num_threads  )

                val_prob, val_stack= layer.train(X_train, y_train)
                
                self.augment_model_temp = layer  


                # temp_val_acc2 = compute_accuracy( y_train, val_prob )
                # print('val acc2:', temp_val_acc2)
                # if temp_val_acc2 > best_train_acc2:
                #     best_train_acc2 = temp_val_acc2
                #     best_test_acc2 = temp_test_acc
                #     best_layer_index2 = layer_index
            ##### END RIT


            X_train_new = np.concatenate([X_train, val_stack], axis=1)
 

            layer_index = layer_index + 1


        del layer
        del self.augment_model_temp

        print( '\nBest layer index: {}'.format(self.best_layer_index) )
        return -1


    
    def predict_proba(self, X_test):
        # TODO
        pass
        # X_test_new = X_test.copy()
        # test_prob = []
        # for layer in self.model:
        #     test_prob, test_stack = layer.predict(X_test_new)
        #     X_test_new = np.concatenate([X_test, test_stack], axis=1)
        #     layer_index = layer_index + 1

        # return test_prob


    def predict(self, X_test):
        print(' {}\n'.format(datetime.datetime.now())  , ':', X_test.shape )
        
        X_test_new = X_test.copy()
        test_prob = []

        if self.best_layer_index > 0:
            
            for layer_index in range( 0, self.best_layer_index  ):
                
                new_features_lst = self.all_layer_features[layer_index]
                bootstrap_feature_importance = self.all_layer_feature_importance[layer_index]

                ####### augment data  #######
                
                ## One feature
                
                new_test_augment = None
                for idx, new_feature in enumerate( new_features_lst ):       
                    
                    test_augment = np.zeros( (X_test.shape[0], ) )

                    for idx2, feature_comp in enumerate(new_feature):
                        
                        temp = X_test[ :, feature_comp['feature_id'] ] - feature_comp[ 'threshold' ]
                        temp = temp * ( bootstrap_feature_importance[ feature_comp['feature_id'] ] * feature_comp['L_R'] )
                        test_augment += temp

                    ### relu
                    test_augment  = np.maximum( test_augment, 0)

                    test_augment = test_augment.reshape(-1,1)

                    ## augment train and test
                    if idx == 0:
                        new_test_augment = test_augment
                    else:
                        new_test_augment = np.concatenate( [new_test_augment, test_augment], axis=1 )
    

                ## augment test
                if ( new_test_augment is not None):
                    X_test = np.concatenate( [X_test, new_test_augment], axis=1 )

            test_prob, test_stack = self.augment_model.predict( X_test )

            X_test_new = np.concatenate([X_test, test_stack], axis=1 )

        test_prob, test_stack = self.predictive_model.predict(X_test_new)
        
        return np.argmax(test_prob, axis=1)






def run_iRF_FPGrowth(X_train,
            X_test,
            y_train,
            y_test,
            rf,
            rf_bootstrap = None,
            initial_weights = None,
            K=7,
            B=10,
            random_state_classifier=2018,
            propn_n_samples=0.2,
            bin_class_type=1,
            min_confidence=0.8,
            min_support=0.1,
            signed=False,
            n_estimators_bootstrap=5,
            bootstrap_num=5):
    """
    Runs the iRF algorithm but instead of RIT for interactions, runs FP-Growth through Spark.


    Parameters
    --------
    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test vector, where n_samples in the number of samples and
        n_features is the number of features.

    y_train : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values for training.

    y_test : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values for testing.

    rf : RandomForest model to fit
    
    rf_bootstrap : random forest model to fit in the RIT stage, default None, which means it is the same as rf.
        The number of trees in this model should be set smaller as this step is quite time consuming.

    K : int, optional (default = 7)
        The number of iterations in iRF.

    n_estimators : int, optional (default = 20)
        The number of trees in the random forest when computing weights.

    B : int, optional (default = 10)
        The number of bootstrap samples

    random_state_classifier : int, optional (default = 2018)
        The random seed for reproducibility.

    propn_n_samples : float, optional (default = 0.2)
        The proportion of samples drawn for bootstrap.

    bin_class_type : int, optional (default = 1)
        ...

    min_confidence: float, optional (default = 0.8)
        FP-Growth has a parameter min_confidence which is the minimum frequency of an interaction set amongst all transactions
        in order for it to be returned
    
    bootstrap_num: float, optional (default = 5)
        Top number used in computing the stability score


    Returns
    --------
    all_rf_weights: dict
        stores feature weights across all iterations

    all_rf_bootstrap_output: dict
        stores rf information across all bootstrap samples

    all_rit_bootstrap_output: dict
        stores rit information across all bootstrap samples

    stability_score: dict
        stores interactions in as its keys and stabilities scores as the values

    """
    # Set the random state for reproducibility
    np.random.seed(random_state_classifier)

    # Convert the bootstrap resampling proportion to the number
    # of rows to resample from the training data
    n_samples = ceil(propn_n_samples * X_train.shape[0])

    # All Random Forest data
    all_K_iter_rf_data = {}

    # Initialize dictionary of rf weights
    # CHECK: change this name to be `all_rf_weights_output`
    all_rf_weights = {}

    # Initialize dictionary of bootstrap rf output
    all_rf_bootstrap_output = {}

    # Initialize dictionary of bootstrap FP-Growth output
    all_FP_Growth_bootstrap_output = {}
    
    if issubclass(type(rf), RandomForestClassifier):
        weightedRF = wrf(**rf.get_params())
    elif issubclass(type(rf) is RandomForestRegressor):
        weightedRF = wrf_reg(**rf.get_params())
    else:
        raise ValueError('the type of rf cannot be {}'.format(type(rf)))
    
    weightedRF.fit(X=X_train, y=y_train, feature_weight = initial_weights, K=K,
                   X_test = X_test, y_test = y_test)
    all_rf_weights = weightedRF.all_rf_weights
    all_K_iter_rf_data = weightedRF.all_K_iter_rf_data

    # Run the FP-Growths
    if rf_bootstrap is None:
            rf_bootstrap = rf
    for b in range(B):

        # Take a bootstrap sample from the training data
        # based on the specified user proportion
        if isinstance(rf, ClassifierMixin):
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples, stratify = y_train)
        else:
            X_train_rsmpl, y_rsmpl = resample(
                X_train, y_train, n_samples=n_samples)

        # Set up the weighted random forest
        # Using the weight from the (K-1)th iteration i.e. RF(w(K))
        rf_bootstrap = clone(rf)
        
        # CHECK: different number of trees to fit for bootstrap samples
        rf_bootstrap.n_estimators=n_estimators_bootstrap

        # Fit RF(w(K)) on the bootstrapped dataset
        rf_bootstrap.fit(
            X=X_train_rsmpl,
            y=y_rsmpl,
            feature_weight=all_rf_weights["rf_weight{}".format(K)])

        # All RF tree data
        # CHECK: why do we need y_train here?
        all_rf_tree_data = get_rf_tree_data(
            rf=rf_bootstrap,
            X_train=X_train_rsmpl,
            X_test=X_test,
            y_test=y_test,
            signed=signed)

        # Update the rf bootstrap output dictionary
        all_rf_bootstrap_output['rf_bootstrap{}'.format(b)] = all_rf_tree_data

        # Run FP-Growth on interaction rule set
        all_FP_Growth_data = generate_all_samples(all_rf_tree_data, bin_class_type)

        spark = SparkSession \
                    .builder \
                    .appName("iterative Random Forests with FP-Growth") \
                    .getOrCreate()
    
        # Load all interactions into Spark dataframe
        input_list = [(i, all_FP_Growth_data[i].tolist()) for i in range(len(all_FP_Growth_data))]
        df = spark.createDataFrame(input_list, ["id", "items"])

        # Run FP-Growth on data
        fpGrowth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
        model = fpGrowth.fit(df)
        item_sets = model.freqItemsets.toPandas()

        # Update the rf_FP_Growth bootstrap output dictionary
        item_sets = item_sets.sort_values(by=["freq"], ascending=False)
        all_FP_Growth_bootstrap_output['rf_bootstrap{}'.format(
            b)] = item_sets

    stability_score = _FP_Growth_get_stability_score(
        all_FP_Growth_bootstrap_output=all_FP_Growth_bootstrap_output, bootstrap_num=bootstrap_num)

    return all_rf_weights,\
        all_K_iter_rf_data, all_rf_bootstrap_output,\
        all_FP_Growth_bootstrap_output, stability_score

def generate_all_samples(all_rf_tree_data, bin_class_type=1):
    n_estimators = all_rf_tree_data['rf_obj'].n_estimators

    all_paths = []
    for dtree in range(n_estimators):
        filtered = filter_leaves_classifier(
            dtree_data=all_rf_tree_data['dtree{}'.format(dtree)],
            bin_class_type=bin_class_type)
        all_paths.extend(filtered['uniq_feature_paths'])
    return all_paths

def _hist_features(all_rf_tree_data, n_estimators,
                   xlabel='features',
                   ylabel='frequency',
                   title='Frequency of features along decision paths'):
    """
    Generate histogram of number of appearances a feature appeared
    along a decision path in the forest
    """

    all_features = []

    for i in range(n_estimators):
        tree_id = 'dtree' + str(i)

        a = np.concatenate(
            all_rf_tree_data[tree_id]['all_uniq_leaf_paths_features'])
        all_features.append(a)

    all_features = np.concatenate(all_features)

    counts = {m: np.sum(all_features == m) for m in all_features}
    data_y = sorted(counts.values(), reverse=True)
    data_x = sorted(counts, key=counts.get, reverse=True)
    plt.figure(figsize=(15, 8))
    plt.clf()
    plt.bar(np.arange(len(data_x)), data_y, align='center', alpha=0.5)
    plt.xticks(np.arange(len(data_x)), data_x, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
