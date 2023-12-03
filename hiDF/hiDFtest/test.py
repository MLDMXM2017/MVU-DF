
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt 
import random
import os

from scipy.sparse import csr_matrix
from functools import reduce

import sklearn
from sklearn import tree
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_boston , \
                            fetch_olivetti_faces, fetch_20newsgroups_vectorized, fetch_covtype , fetch_rcv1, fetch_20newsgroups, \
                            make_spd_matrix

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn import preprocessing, manifold, datasets


from hiDF.tree import WeightedExtraTreeClassifier, WeightedDecisionTreeClassifier
from hiDF.ensemble import RandomForestRegressorWithWeights, RandomForestClassifierWithWeights, ExtraTreesClassifierWithWeights

from hiDF.hiDF_jupyter_utils import draw_tree
from hiDF.utils import all_tree_signed_threshold_paths , get_tree_data, get_rf_tree_data, get_marginal_importances
from hiDF.hiDF_utils import  build_tree 
from hiDF.hiDF_utils_threshold import  generate_rit_samples, build_tree_threshold ,\
                                     get_rit_tree_data, rit_interactions, _get_stability_score , \
                                     run_hiDF, gcForest_hi, gcForest_hi_new
                                    
from hiDF import hiDF_utils
from hiDF.utils import t_sne_visual, plot2d, plot3d, plot2d_2color, plot_t_sne_from_file

import warnings
from sklearn.inspection import permutation_importance

#import lightgbm as lgb

try:
    from xgboost import XGBClassifier
except ImportError:
    warnings.warn('No xgboost installed')


try:
    from lightgbm import LGBMClassifier 
except ImportError:
    warnings.warn('No lightgbm installed')



from hiDF.gcForest import gcForest1



my_leaf_paths = []


def RIT_test( all_rf_tree_data, bin_class_type=None, max_depth=3, noisy_split=False, num_splits=4 ):
    '''
    hiDF_utils.py / get_rit_tree_data()
    '''

    # Create the weighted randomly sampled paths as a generator
    gen_random_leaf_paths = generate_rit_samples(
        all_rf_tree_data=all_rf_tree_data,
        bin_class_type=bin_class_type)


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
    rit_leaf_node_union_value = reduce(np.union1d, rit_leaf_node_values)
    rit_output = {"rit": rit,
                    "rit_intersected_values": rit_intersected_values,
                    "rit_leaf_node_values": rit_leaf_node_values,
                    "rit_leaf_node_union_value": rit_leaf_node_union_value}
    
    print('\n------ RIT output ------\n\n')
    for i in rit_output:
        print('\n', i , '\n', rit_output[i] )

    print('\n-----END ----- \n')



def rf_structure( X_train , y_train, X_test, y_test ):
    ''' 
    '''

    
    NEED_CHECK = False
    if NEED_CHECK:
        print('\nFit RandomForestClassifierWithWeights')
        rf = RandomForestClassifierWithWeights( n_estimators=2, random_state=2019)
        rf.fit(X=X_train, y=y_train )

        all_rf_tree_data = get_rf_tree_data(
                                rf=rf, X_train=X_train, X_test=X_test, y_test=y_test, signed=True, threshold=True)
        
        print('\nTest RIT Tree:  get_rit_tree_data \n')
        all_rit_tree_data = get_rit_tree_data( all_rf_tree_data, bin_class_type=None )
        for key in all_rit_tree_data.keys():
            print(key, ' : \n', all_rit_tree_data[key] , '\n')

    ## my test RIT
    ##RIT_test(all_rf_tree_data)

    ### CHECK GOOD!
    NEED_CHECK = False
    if NEED_CHECK:
        print('\n-----------------------')
        print('Test hiDF_utils_threshold.py / rit_interactions()\n One run of RIT (M RIT trees)')
        interactions_threshold = rit_interactions(all_rit_tree_data)
        for key in interactions_threshold :
            print(key, '\n', interactions_threshold[key] , '\n' )
        print('One Run RIT END -------------\n')


        ## Test bootstrap GOOD
        all_rit_bootstrap_output = {}
        for i in range(10):
            ## bootstrap
            X_train_rsmpl, y_rsmpl = resample( X_train, y_train, n_samples=0.6* X_train.shape[0] , stratify = y_train)
            rf.fit(X=X_train_rsmpl, y=y_rsmpl )
            all_rf_tree_data = get_rf_tree_data( rf=rf, X_train=X_train_rsmpl, X_test=X_test, y_test=y_test, signed=True, threshold=True)

            all_rit_tree_data = get_rit_tree_data( all_rf_tree_data, bin_class_type=None )
            all_rit_bootstrap_output['rf_bootstrap{}'.format(i)] = all_rit_tree_data

            if i == 0:
                print('\n\n-------------\nFirst RIT interactions in bootstrap\n\n')
                interactions_threshold = rit_interactions(all_rit_tree_data)
                for key in interactions_threshold :
                    print(key, '\n', interactions_threshold[key] , '\n' )

        bootstrap_interact_stability, bootstrap_interact_threshold = _get_stability_score( all_rit_bootstrap_output )
        print('\n-----------------\nstability score:\n')
        for k in sorted (bootstrap_interact_stability) : 
            print (k, ' , ', bootstrap_interact_stability[k], ' \n    threshold:', bootstrap_interact_threshold[k] ,'\n')

        sorted_stability_lst = sorted( bootstrap_interact_stability.items(), key = lambda kv:(kv[1], kv[0]), reverse=True ) 
        print('\nstability\n')
        for kv in sorted_stability_lst:
            print( kv[0], ' , stability: ' , kv[1], ' \n    threshold:', bootstrap_interact_threshold[kv[0]] ,'\n' )
            
    print('\n\n----- rf RIT end -----\n\n')

    ## CHECK GOOD!  
    ## generate_rit_samples : weights and paths are correct!
    NEED_CHECK = False
    if( NEED_CHECK ):
        print( '\n--------------\nTest hiDF_utils generate_rit_samples , weights and values...')
        
        _, (all_paths, all_weights) = generate_rit_samples( all_rf_tree_data, bin_class_type=None, return_path_and_weights=True )

        for i,j in zip(all_paths, all_weights):
            print( '\npath: ' , i , '\nweight: ', j )


        for idx, estimator in zip( range( len(rf.estimators_) ) , rf.estimators_ ):
            print('-----------\n{} th tree'.format(idx) )
            print( '\ntree_.value\n' , estimator.tree_.value )
            print( '\ntree node, child nodes:\n' ,estimator.tree_.children_left , '\n' , estimator.tree_.children_right )
            print( '\ntree feature:\n', estimator.tree_.feature )
            print( '\ntree threshold value:\n', estimator.tree_.threshold )

            # print('\nUtils: all_tree_signed_threshold_paths test \n')
            # paths = all_tree_signed_threshold_paths(estimator, 0)
            # for path in paths:
            #     print(path)
            # print('\n')
            
            print('\nIn utils.py , get_tree_data() \n')
            tree_data = get_tree_data(X_train,X_test,y_test, estimator, 0, signed=True, threshold=True )
            for i in tree_data:
                print(i ,'\n', tree_data[i] )
            print('\n--------------------\n\n')



def tree_structure(X_train , y_train, X_test, y_test):
    

    print('Train data shape: ', X_train.shape )

    estimator = DecisionTreeClassifier(random_state=1) #(max_leaf_nodes=3, random_state=0)
    # estimator = WeightedExtraTreeClassifier(random_state=1, max_leaf_nodes=5)
    estimator.fit(X_train, y_train)
    

    ##########################################
    ####  test utils.py  ###############
    print('\ntree maxdepth:\n', estimator.tree_.max_depth, '\n')
    print( '\ntree_.value\n' , estimator.tree_.value ,'\n\n')
    print( '\ntree node, child nodes:\n' ,estimator.tree_.children_left , '\n' , estimator.tree_.children_right , '\n')
    print( '\ntree feature:\n', estimator.tree_.feature ,'\n')
    print( '\ntree threshold value:\n', estimator.tree_.threshold , '\n' )

    print('\nUtils: all_tree_signed_threshold_paths test \n')
    paths = all_tree_signed_threshold_paths(estimator, 0)
    for path in paths:
        print(path)
    print('\n')
    
    print('\nIn utils.py , test get_tree_data() \n')
    tree_data = get_tree_data(X_train,X_test,y_test, estimator, 0, signed=True, threshold=True )
    for i in tree_data:
        print(i ,'\n', tree_data[i] )
    print('\n--------------------\n\n')



    ######################################################
    ## 
    ## https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#understanding-the-decision-tree-structure

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
        "the following tree structure:"
        % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                "node %s."
                % (node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    ))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
            % (node_id,
                sample_id,
                feature[node_id],
                X_test[sample_id, feature[node_id]],
                threshold_sign,
                threshold[node_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
        % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

    return estimator



### adult / yeast / letter
def load_my_data(dataset):
    ''' adult / letter / yeast data
    '''

    path = "./dataset/{}/".format(dataset)
    train_data = np.loadtxt(path+'train.txt', skiprows=1 )  
    train_label = np.loadtxt(path+'label_train.txt')  
    test_data = np.loadtxt(path+'test.txt', skiprows=1 )  
    test_label = np.loadtxt(path+'label_test.txt')
    return [train_data,train_label,test_data,test_label]


### my load svm
def load_svm(file, feature_num):
    Matrix = []
    targets = []
    with open(file) as f:
        for line in f:
            ## Get all variable-length spaces down to two. Then use two spaces as the delimiter.
            while line.replace("   ", "  ") != line:
                line = line.replace("   ", "  ")
            
            data = line.split('  ')
            target = float(data[0]) # target value
            targets.append( target )
            row = []

            all_lst = [item.strip().split(':') for item in data[1:]]
            last_index = 1
            for i, (idx, value) in enumerate(all_lst):
                n = int(idx) - last_index # num missing
                for _ in range(n-1):
                    row.append(0) # for missing
                
                last_index = int(idx)

                row.append(float(value.strip()))
            temp_length = len(row)
            for i in range(temp_length, feature_num):
                row.append(0)
            
            Matrix.append( row )

    #print(len(Matrix), len(Matrix[0]), len(Matrix[-1]))
    res = np.array(Matrix)
    y = np.array(targets, np.int32 )
    
    return  res, y



def load_svm2(file, feature_num):
    '''： Gas Sensor Array Drift Dataset at Different Concentrations Data Set
    https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
    '''
    Matrix = []
    targets = []
    with open(file) as f:
        for line in f:
            ## Get all variable-length spaces down to two. Then use two spaces as the delimiter.
            while line.replace("  ", " ") != line:
                line = line.replace("  ", " ")
            
            data = line.split(' ')
            if( data[-1]=='\n'):
                data = data[:-1]
            
            target = float(data[0]) # target value
            targets.append( target )
            row = []

            all_lst = [item.strip().split(':') for item in data[1:]]
            last_index = 1
            for i, (idx, value) in enumerate(all_lst):
                n = int(idx) - last_index # num missing
                for _ in range(n-1):
                    row.append(0) # for missing
                
                last_index = int(idx)

                row.append(float(value.strip()))
            temp_length = len(row)
            for i in range(temp_length, feature_num):
                row.append(0)
            
            Matrix.append( row )

    #print(len(Matrix), len(Matrix[0]), len(Matrix[-1]))
    res = np.array(Matrix)
    y = np.array(targets, np.int32 )
    
    return  res, y




def my_one_hot_numpy1( X , feature_ids):
    df = pd.DataFrame(X)
    #print( df_train[:3] )
    for idx in feature_ids:
        df = pd.concat([df, pd.get_dummies(df[idx], prefix='{}_'.format(idx))], axis=1)
        df.drop([idx],axis=1, inplace=True)

    return df.to_numpy()



def my_one_hot_numpy( X, X_test , feature_ids ):
    '''
    Parameters
    ----------
    X, X_test : 
        numpy 2D array   
    feature_ids: 
        list of int

    return
    ------
    X, X_test:
        numpy 2D array 
    '''
    #print('my one hot')
    df_train = pd.DataFrame(X)
    df_test = pd.DataFrame(X_test)

    #print( df_train[:3] )
    for idx in feature_ids:
        df_train = pd.concat([df_train, pd.get_dummies(df_train[idx], prefix='{}_'.format(idx))], axis=1)
        df_train.drop([idx],axis=1, inplace=True)
        #print( df_train[:3] )
        df_test = pd.concat([df_test, pd.get_dummies(df_test[idx], prefix='{}_'.format(idx))], axis=1)
        df_test.drop([idx],axis=1, inplace=True)

    return df_train.values, df_test.values


def my_one_hot_pandas( df , cols ):

    for col in cols:
        df = pd.concat([df, pd.get_dummies(df[col], prefix='{}_'.format(col))], axis=1)
        df.drop([col],axis=1, inplace=True)
        
    return df



from numpy.random import multivariate_normal, standard_cauchy, standard_normal
from pandas import DataFrame
from matplotlib import pyplot

def simulation_data():
    '''
    

    '''
    has_test=False

    X, X_test, y, y_test = None, None, None, None

    

    pass
    
    threshold_or = 3.2
    threshold_and = -1
    threshold_xor = 1



    



    ##################################################
    ##################################################
    ###### sklearn make circle #######################
    ##################################################

    if True:

        print('\n\nSklearn make circle, T-SNE visual \n\n')

        ## -----------------------------------------------------------------------------------
            ## T-SNE config:
            ## num_instances=10000， num_noisy_features=500， circle(factor : 0.35, noise: 0.05)， noise: -0.5 ~ +0.5， new_feature_limit=10
            ##
            ## ------------------------------------------------------------------------------------


        num_instances = 10000  ## 10000 

        num_noisy_features = 500  ## 500
        X, y = datasets.make_circles( n_samples= num_instances, factor=0.35, noise=.05, random_state=42 )
        #print( np.mean(X[:,0]) , np.max(X[:,0]), np.min(X[:,0]) , np.mean(X[:,1]) , np.max(X[:,1]), np.min(X[:,1]) )



        ####  plot2d
        # plot2d_2color( X, y )
        

        ## feature
        if num_noisy_features > 0:  ## -0.5 ~ +0.5
                X_temp = np.random.uniform( -0.5, 0.5 , size=( len(X), num_noisy_features) )
                X = np.concatenate([X,X_temp], axis=1 )


        
        has_test = False
        stability_threshold = 0.2
        new_feature_limit = 10   ## 10


    

    ###############################
    ##########  !!!!!  ############
    if False:
        
        print('\n\n  \n\n')

        n_samples = 1000
        range_end = 10
        step_num = 5

        np.random.seed(42)
        X = np.random.uniform( 0, range_end, size=( n_samples, 2 ) )
        y = np.zeros( n_samples )

        ##################
        ### 
        index = (X[:,0] == 0) | (X[:,0]==range_end) | (X[:,1]==0) | (X[:,1] == range_end)
        X = np.delete(X, index, axis=0)
        y = np.delete(y, index)


        
        
        ############################################################
        ######  XOR   ###################################
        ############################################################
        
        X = X - [5,5]

        index = ( (X[:, 0] + X[:, 1]) >= 0  ) &  ( ( X[:, 1] - X[:, 0] ) >= 0 )
        index = index |  ( (X[:, 0] + X[:, 1]) < 0  ) &  ( ( X[:, 1] - X[:, 0] ) < 0 )
        
        y[index] = 1
        

        # # #############
        # X_interact = None
        # X_temp =  (X[:,0] - 0) + (X[:,1] - 0) - 0  ## np.maximum( (X[:,0] - 0) + (X[:,1] - 0) - 0 , 0 )
        # X_temp = X_temp.reshape(-1,1)
        # X_interact = X_temp
        # X_temp = -1*(X[:,0] - 0) + (X[:,1] - 0) - 0  ## np.maximum( -1*(X[:,0] - 0) + (X[:,1] - 0) , 0 )
        # X_temp = X_temp.reshape(-1,1)
        # X_interact = np.concatenate( [X_interact, X_temp], axis=1 )
        
        # #### 
        # X = X_interact
        
        # plot2d( X, y, file_name='oblique_XOR')
        df = pd.DataFrame( X[:, -2: ], columns=['comp1', 'comp2'])
        df['label'] = y
        
        import seaborn as sns
        import matplotlib.colors as mcolors
        
        fig = plt.figure( figsize=( 5.5 , 5.5 )  )   
        ax1 = plt.subplot( 1, 1, 1 )
        # plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        
        ## option1
        temp = sns.lmplot(x='comp1', y='comp2', data=df, hue='label', fit_reg=False, markers=['o','x'],scatter_kws={"s": 50}, palette=[ mcolors.to_rgb('slategrey') ],  legend=False  )    ## , palette=sns.color_palette('deep', n_color=1)     ,scatter_kws={"facecolors" : 'None'}  

        temp.fig.set_size_inches( 5.7, 5 ) 

        
        
        plt.xlabel( None )   #  r"$X_1$", fontsize=18
        plt.ylabel( None )
        plt.xticks([-10, -5, 0, 5, 10], fontsize = 13)  
        plt.yticks([-10, -5, 0, 5, 10], fontsize = 13) # (fontsize = 13)


        # X, y = datasets.make_circles(n_samples= 2000, factor=.35, noise=.05, random_state=42)
        
        

        X, X_test, y, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
        mytree = tree_structure(X, y, X_test, y_test)


        


        ################################################
        ########  Plot decision tree boundary ##########
        ################################################
        
        n_classes = 2
        plot_colors = "rb"
        plot_step = 0.02

        clf = DecisionTreeClassifier().fit(X, y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))
        
        

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        cs = plt.contour(xx, yy, Z, colors='orangered' , linewidths=2.0, linestyles='dashed' )
         

        

        ## Plot the training points
        # for i, color in zip(range(n_classes), plot_colors):
        #     idx = np.where(y == i)
        #     plt.scatter(X[idx, 0], X[idx, 1], c=color, label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

        plt.savefig( './fig/tree_{}.pdf'.format( int(time.time() ) ) ,  bbox_inches='tight', pad_inches=0.02, format='pdf')
        
        ###################################################

        
        
        input('xxxxx  END  xxxxx')





    
    # print(X[:10], y[:10])
    print( np.unique(y, return_counts=True) )
    # input('xxxxx')

    
    le = preprocessing.LabelEncoder()
    y = le.fit_transform( y )

    if has_test:
        y_test = le.transform( y_test )

    
    return  has_test, X, X_test, y, y_test, stability_threshold, new_feature_limit
    



from sklearn.datasets import load_svmlight_file
import scipy.io as sio
from scipy.io import arff




def read_raw_data( random_state=None , IF_standardization=False ):
    
    
    has_test=True

    X, X_test, y, y_test = None, None, None, None


    new_feature_limit = 5
    stability_threshold = 0.5


    #############################################################
    ################### sklearn toy dataset #####################
    #############################################################
        ## iris data: multi class 3 class ( 150 samples * 4 features)
        #sklearn_data = load_iris()
        
        ## breast cancer data : binary classification  (569 samples * 30 features)
        #sklearn_data = load_breast_cancer()
        
        ## digit data : 10 class classification ( 1797 samples * 64 features )
        #sklearn_data = load_digits()

        ## boston house price, regression ( 506 samples * 13 features )
        #sklearn_data =  load_boston()

    #############################################################
    ################### sklearn real dataset ####################
    #############################################################

        # fetch_olivetti_faces 
        # sklearn_data = fetch_olivetti_faces() 
        # stability_threshold = 0.1


        ######
        #####  news data , shape: (18846, 130107)  // no new features generated
        
            # print('Loading 20newsgroups data...')
            # # sklearn_data = fetch_20newsgroups_vectorized(subset='all')
            
            # vectorizer = TfidfVectorizer()

            # X, y = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes') , return_X_y=True)
            # X, y = X[:10000], y[:10000]
            # X = vectorizer.fit_transform(X)
            # X, y = X.toarray(), y

            # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes') )
            # X_test = vectorizer.transform(newsgroups_test.data)
            # y_test = newsgroups_test.target
            # X_test, y_test = X_test.toarray(), y_test
            
            # stability_threshold = 0.2
            # new_feature_limit = 50



        #########################################
        ### sklearn cover type data , shape:( 581012 * 54 )
        # print('Loading cover type data...')
        # sklearn_data = fetch_covtype()
        # stability_threshold = 0.5


        # X = sklearn_data.data
        # y = sklearn_data.target
        # has_test = False



    ##########################################################################################
    ########### libsvm dataset ###############################################################
    ##########################################################################################

        

        ### GOOD   

            # print('\nLoading libsvm covtype dataset ; 7-class (581012*54)')
            # X, y = load_svmlight_file('./dataset/libsvm_data/covtype/covtype')
            # X = X.toarray()
            # has_test = False

            # # ## Attention!!
            # # X, X_test, y, y_test = train_test_split(X, y, random_state=42, train_size=50000 , shuffle=True, stratify= y)            
            # # has_test = True
            
            # stability_threshold = 0.3
            # new_feature_limit = 5



        
        ## GOOD ;
            

            # print('\nLoading libsvm covtype binary dataset ; (581012*54)')
            # X, y = load_svmlight_file('./dataset/libsvm_data/covtype/covtype.libsvm.binary')
            # X = X.toarray()
            # has_test = False
            # stability_threshold = 0.5
            # new_feature_limit = 5




        


    ##########################################################################################
    ############# UCI dataset ################################################################
    ##########################################################################################

        
        ### GOOD , 2020-11-20
            
            # print('\nLoading UCI "default of credit card clients Data Set" , 2-classification, (30000 * 24 )')
            # data = pd.read_excel('./dataset/uci_data/credit_card/default of credit card clients.xls', skiprows=1 )
            # data = my_one_hot_pandas( data, ['SEX', 'EDUCATION', 'MARRIAGE'])
            # y= data['default payment next month'].to_numpy()
            # data.drop(['default payment next month', 'ID'],axis=1, inplace=True)  
            # X = data.values
            # has_test = False
            # stability_threshold = 0.2
            # new_feature_limit = 5  ## 4 


        
        ### GOOD
        

            # print('\nLoading UCI "YearPredictionMSD Data Set" , regression/classification, (515345 * 90 )')
            # data = genfromtxt('./dataset/uci_regression/year_prediction/YearPredictionMSD.txt', delimiter=',' )
            
            # X = data[:, 1:] 
            # y = data[:, 0].astype(np.int32)
            
            # # index = np.argwhere(y==2011)  
            # # X = np.delete(X, index, axis=0)
            # # y = np.delete(y, index)

            # # index = np.argwhere(y<=1980)  
            # # X = np.delete(X, index, axis=0)
            # # y = np.delete(y, index)

            # # y_unique, y_counts = np.unique(y, return_counts=True) 
            # # for idx , year in enumerate( y_unique ):
            # #     print( year ,' ：' , y_counts[idx] )
            # # print( np.unique(y, return_counts=True) )
            # # input('xx')
            
            # # y = y/5
            # # y = y.astype(int)

            # ### 2000 binary classification
            # y = y < 2000   

            # if False:
            #     y[ np.nonzero( (y>1920)&(y <= 1930) ) ] = 0
            #     y[ np.nonzero( (y>1930)&(y <= 1940) ) ] = 1
            #     y[ np.nonzero( (y>1940)&(y <= 1950) ) ] = 2
            #     y[ np.nonzero( (y>1950)&(y <= 1960) ) ] = 3
            #     y[ np.nonzero( (y>1960)&(y <= 1970) ) ] = 4
            #     y[ np.nonzero( (y>1970)&(y <= 1980) ) ] = 5
            #     y[ np.nonzero( (y>1980)&(y <= 1990) ) ] = 6
            #     y[ np.nonzero( (y>1990)&(y <= 2000) ) ] = 7
            #     y[ np.nonzero( (y>2000)&(y <= 2010) ) ] = 8
            # has_test = False
            # stability_threshold = 0.2
            # new_feature_limit = 5

            


        # ### GOOD
            # print('\nLoading UCI "Arrhythmia Data Set" , classification, (452 * 279 )')
            # data = genfromtxt('./dataset/uci_life/arrhythmia/arrhythmia.data', delimiter=',',dtype=float, missing_values='?', filling_values=0.0 )
            # X = np.copy( data[:, :-1] )
            # y = np.copy( data[:,-1] )
            # y = y.astype(np.int32)
            
            # has_test = False
            # stability_threshold = 0.3
            # new_feature_limit=10
                            


    
        
        ### GOOD 

            # print('\nLoading UCI "Bank Marketing Data Set" , classification, (45211 * 17/20 )')
            # data = pd.read_csv('./dataset/uci_data/Bank Marketing/bank-additional-full.csv', sep=';', na_values="unknown")
            # # print(data.info)
            # # print('\n----------\n',data[:5])
            # # print('\nNan:\n', data.isna().sum() )
            # data = my_one_hot_pandas( data, ['job','marital','education','default','housing','loan','contact', 'month','day_of_week', 'poutcome'])
            # y = data['y'].to_numpy(copy=True)
            # X = data.drop(['y', 'duration'], axis=1, inplace=False).to_numpy()
            # has_test = False
            # stability_threshold = 0.3
            # new_feature_limit = 5



        ### GOOD 
            
            # print('\nLoading UCI "Crowdsourced Mapping Data Set" , classification, (10546 * 29 )')
            # train_data = genfromtxt('./dataset/uci_data/Crowdsourced Mapping/training.csv', dtype=str, skip_header=1, delimiter=',' )
            # test_data = genfromtxt('./dataset/uci_data/Crowdsourced Mapping/testing.csv', dtype=str, skip_header=1, delimiter=',' )
            # X = train_data[:,1:].astype(float)
            # y = train_data[:,0]
            # # print(X[:3], y[:3])
            # X_test = test_data[:,1:].astype(float)
            # y_test = test_data[:,0]
            
            # stability_threshold = 0.5
            # new_feature_limit = 3



        

        ### GOOD ; 2020-11-21
            # hiDF  0.622326
            # hiDF+gcForest  0.6214941
            # gcForest  62.26316
            # gcForest_HI  62.43237 ！！！
            # xgboost  0.6099675
            # lightgbm  0.627541

            # print('\nLoading UCI "Diabetes 130-US hospitals for years 1999-2008 Data Set" , classification, (10000*55 )')        
            # df = pd.read_csv('./dataset/uci_data/diabetes_130_us/diabetic_data.csv')
            # df.replace('?',np.nan,inplace=True)
            # #print( df.head() )
            # #dropping columns with high NA percentage
            # df.drop(['weight','medical_specialty','payer_code'],axis=1,inplace=True)
            # # dropping columns related to IDs
            # df.drop(['encounter_id','patient_nbr','admission_type_id',
            #         'discharge_disposition_id','admission_source_id'],axis=1,inplace=True)
            # #removing invalid/unknown entries for gender
            # #print(len(df))
            # df=df[df['gender']!='Unknown/Invalid']
            # #print(len(df))
            # df.dropna(inplace=True)
            # #print( df['diag_3'][:10] , df['diag_3'][1]=='255' , df['diag_3'][2] )
            # df = df[df.diag_1.apply(lambda x: x.isnumeric())]
            # df = df[df.diag_2.apply(lambda x: x.isnumeric())]
            # df = df[df.diag_3.apply(lambda x: x.isnumeric())]
            # #print( df['diag_3'][:10]  )
            # df["diag_1"] = pd.to_numeric(df["diag_1"], downcast="float")
            # df["diag_2"] = pd.to_numeric(df["diag_2"], downcast="float")
            # df["diag_3"] = pd.to_numeric(df["diag_3"], downcast="float")

            # cat_cols = list(df.select_dtypes('object').columns)
            # ##  label
            # cat_cols = cat_cols[:-1]
            # print( cat_cols )
            # class_dict = {}
            # for col in cat_cols:
            #     df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col])], axis=1)
            
            # X = df.drop('readmitted',axis=1).to_numpy().astype(float)
            # df['readmitted'] = df['readmitted'].replace('>30', 1)
            # df['readmitted'] = df['readmitted'].replace('<30', 1)
            # df['readmitted'] = df['readmitted'].replace('NO', 0)
            # y = df['readmitted'].to_numpy().astype(int)
            
            # has_test = False
            # stability_threshold = 0.3
            # new_feature_limit = 5


        
        ### GOOD 
            # print( '\nLoading UCI "Statlog (Landsat Satellite) Data Set"  (libsvm satimage.scale),  (6435*36) ' )
            # train = np.loadtxt('./dataset/uci_data/statlog_satellite/sat.trn', dtype=float )
            # test = np.loadtxt('./dataset/uci_data/statlog_satellite/sat.tst', dtype=float )
            # X, y = train[:, 0:-1], train[: , -1].astype(int)
            # X_test, y_test = test[:, 0:-1], test[: , -1].astype(int)       
            # stability_threshold = 0.2
            # new_feature_limit = 5


       

    ################################################
    ####### adult
    
    #'''

    print('\nLoading adult data')
    X, y, X_test, y_test = load_my_data('adult')
    stability_threshold = 0.5
    new_feature_limit = 6
    #X , y = X[:5000], y[:5000]
    use_adult = True 
    if use_adult:
        temp = np.where(X[:,13]==41)
        X[temp[0][0], 13] = 0

        # X = X[:, [0,2,4,10,11,12]]
        # X_test = X_test[:, [0,2,4,10,11,12]]
        
        X, X_test = my_one_hot_numpy(X, X_test, [1,3,5,6,7,8,9,13])
        
        manual_one_hot_adult = False
        if manual_one_hot_adult:
            ### 对adult categorical variable one-hot encoding
            df_train = pd.DataFrame(X)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[1], prefix='workclass')], axis=1)
            df_train.drop([1],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[3], prefix='edu')], axis=1)
            df_train.drop([3],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[5], prefix='marital')], axis=1)
            df_train.drop([5],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[6], prefix='occupation')], axis=1)
            df_train.drop([6],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[7], prefix='relationship')], axis=1)
            df_train.drop([7],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[8], prefix='race')], axis=1)
            df_train.drop([8],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[9], prefix='sex')], axis=1)
            df_train.drop([9],axis=1, inplace=True)
            df_train = pd.concat([df_train, pd.get_dummies(df_train[13], prefix='country')], axis=1)
            df_train.drop([13],axis=1, inplace=True)

            df_test = pd.DataFrame(X_test)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[1], prefix='workclass')], axis=1)
            df_test.drop([1],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[3], prefix='edu')], axis=1)
            df_test.drop([3],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[5], prefix='marital')], axis=1)
            df_test.drop([5],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[6], prefix='occupation')], axis=1)
            df_test.drop([6],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[7], prefix='relationship')], axis=1)
            df_test.drop([7],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[8], prefix='race')], axis=1)
            df_test.drop([8],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[9], prefix='sex')], axis=1)
            df_test.drop([9],axis=1, inplace=True)
            df_test = pd.concat([df_test, pd.get_dummies(df_test[13], prefix='country')], axis=1)
            df_test.drop([13],axis=1, inplace=True)

            X = df_train.values
            X_test = df_test.values
    
    #'''


    

    
    ##### Standardization ######
    IF_standardization = False
    # IF_standardization = True
    if IF_standardization:
        print( '\nRaw Data standardization...')
        # print('\n{}\nData inspect\nX_train'.format('-'*40) )
        # for i in range(X.shape[1]):
        #     print('{}, min{}, max{}, mean:{}, std:{}'.format( i , np.min(X[:,i]) , np.max(X[:,i]), np.mean(X[:,i]), np.std(X[:,i])  ))

        scaler  = preprocessing.StandardScaler().fit(X)  
        X = scaler.transform(X)
        if has_test:
            X_test = scaler.transform(X_test)
        
        # print('\n{}\nData inspect\nX_train'.format('-'*40) )
        # for i in range(X.shape[1]):
        #     print('{}, min{}, max{}, mean:{}, std:{}'.format( i , np.min(X[:,i]) , np.max(X[:,i]), np.mean(X[:,i]), np.std(X[:,i])  ))
        # input('Std.......')


    ########  Label encoder  ########
    le = preprocessing.LabelEncoder()
    y = le.fit_transform( y )

    if has_test:
        y_test = le.transform( y_test )

    
    return  has_test, X, X_test, y, y_test, stability_threshold, new_feature_limit

 


def resample_data( IF_resample, X, y, new_data_num=2000 , random_state=None ):
    '''
    
    
    '''

    X1 , y1 = X, y

    if IF_resample:
        X1, y1 = resample( X, y , replace=False, n_samples=new_data_num, random_state=random_state ) #, stratify=y )

        
        unique_y , counts = np.unique( y1, return_counts=True)
        for i, count in enumerate( counts ):
            if( count == 1 ):
                index = np.argwhere( y1==unique_y[i] )  
                X1 = np.delete(X1, index, axis=0)
                y1 = np.delete(y1, index)

        le = preprocessing.LabelEncoder()
        y1 = le.fit_transform( y1 )


    return X1, y1



def resample_train( IF_resample, X, y, new_train_num=2000 , random_state=None ):
    '''
    
    Parameters:
    ----------

    IF_resample: bool
        

    X: 
        train data

    y:
        train label
    '''

    X1 , y1 = X, y

    if IF_resample:
        X1, y1 = resample( X, y , replace=False, n_samples=new_train_num, random_state=random_state, stratify=y )
    
    return X1, y1





def hiDF_test():

    if False:
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        x = csr_matrix((data, indices, indptr), shape=(3, 3))

        print( x.toarray(),'\n',  x.indices , '\n', x.data, '\n', x.indptr )
        print('\n', x.indices[ x.indptr[0] : x.indptr[1] ] )


    has_test, X_raw, X_test, y_raw, y_test, stability_threshold, new_feature_limit  = simulation_data()
    print("simulation data done...\n")


    random_state = 42 
    
     
    ### 
    start_random, end_random = 20, 25


    ###########  Read Raw Data ###########
    ####
    has_test, X_raw, X_test, y_raw, y_test, stability_threshold, new_feature_limit = read_raw_data()    # load adult dataset
    print( '  Raw data shape: {}'.format(X_raw.shape) )
    print( '  This is a {} classification, min label:{}, max_label:{}'.format( len(np.unique(y_raw)) , np.min(y_raw), np.max(y_raw) ) )
    print( '  Label instance num: ', np.unique(y_raw, return_counts=True ) )
    #print( np.unique(y_raw, return_counts=True) )

    
    ### big data resample    
    IF_resample_all = False
    # IF_resample_all = True
    resample_num_all = 50000

    IF_resample_train = False
    # IF_resample_train = True
    resample_num_train= 10000



    

    if_run_hidf = False
    if_run_hidf_gcf = False
    run_gcForest_HI = False
    run_gcForest = False
    run_xgboost = False
    run_lightgbm = False
    run_gradientboosting = False
    run_RandomForest = False 
    run_LinearSVC = False
    run_RBF_SVC = False


    ### Attention
    metric_all = 'acc'  ##  'acc' / 'roc_auc'  / 'f1'


    # if_run_hidf = True
    # if_run_hidf_gcf = True

    # run_gcForest = True
    run_gcForest_HI = True
    
    run_xgboost = True
    # run_lightgbm = True

    # run_gradientboosting = True
    # run_RandomForest = True
    # run_LinearSVC = True
    # run_RBF_SVC = True




    

   
    ##########################################################
    ##################   gcForest  ###########################
    ##########################################################

    #### gcForest , gcForest_HI , hyper parameters
    num_estimator = 100
    num_forests = 5
    RIT_bin_class_type = None

    
    if run_gcForest:
        performances = []

        if has_test:
            for random_state in range(start_random, end_random):
            
                print('\n{}\n gcForest random:{}\n{}'.format('#'*60, random_state, '#'*60 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                print('{}\nRun gcForest...\n{}'.format('-'*60, '-'*60) )
                
                clf = gcForest_hi(num_estimator=num_estimator, num_forests=num_forests, num_classes=len(np.unique(y)), max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1 , \
                                    sample_weight=None, random_state=random_state, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ,
                                    use_RIT=False , use_metric= metric_all )

                best_test_acc = clf.train( X, y, X_test, y_test )

                # print( datetime.datetime.now(), 'Start prediction...')
                # y_pred = clf.predict(X_test)
                # best_test_acc2 = accuracy_score(y_test, y_pred)
                
                performances.append( best_test_acc )

            pass

        else:
            
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold

            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n gcForest kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                print('{}\nRun gcForest...\n{}'.format('-'*60, '-'*60) )
                
                clf = gcForest_hi(num_estimator=num_estimator, num_forests=num_forests, num_classes=len(np.unique(y)), max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1 , \
                                    sample_weight=None, random_state=random_state, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ,
                                    use_RIT=False, use_metric= metric_all )

                best_test_acc = clf.train( X, y, X_test, y_test )
                performances.append( best_test_acc )

            pass


        print( '\n{}  test acc \n {}\n'.format( datetime.datetime.now(), np.mean(performances,axis=0) ) )
        



    ##########################################################
    ##################  gcForest + HI ########################
    ##########################################################
    
    
    ### RIT class type
    RIT_bin_class_type = None


    if run_gcForest_HI:
        performances = []

        if has_test:
            for random_state in range(start_random, end_random):
            
                print('\n{}\n gcForest_HI random:{}\n{}'.format('#'*60, random_state, '#'*60 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                print('{}\nRun gcForest_HI...\n{}'.format('-'*60, '-'*60) )
                
                clf = gcForest_hi(num_estimator=num_estimator, num_forests=num_forests, num_classes=len(np.unique(y)), max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1 , \
                            sample_weight=None, random_state=random_state, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ,
                            use_metric= metric_all,  use_RIT=True, new_feature_limit=new_feature_limit, stability_threshold=stability_threshold , bin_class_type=RIT_bin_class_type )

                best_test_acc = clf.train( X, y, X_test, y_test )
                performances.append( best_test_acc )

            pass

        else:
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold
            
            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n gcForest_HI kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                print('{}\nRun gcForest_HI...\n{}'.format('-'*60, '-'*60) )
                
                clf = gcForest_hi(num_estimator=num_estimator, num_forests=num_forests, num_classes=len(np.unique(y)), max_layer=10, max_depth=None, n_fold=5, min_samples_leaf=1 , \
                            sample_weight=None, random_state=random_state, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ,
                            use_metric= metric_all, use_RIT=True, new_feature_limit=new_feature_limit, stability_threshold=stability_threshold , bin_class_type=RIT_bin_class_type )

                best_test_acc = clf.train( X, y, X_test, y_test )
                performances.append( best_test_acc )

            pass


        print('\ngcForest_HI ：\n {}\n'.format( np.mean(performances,axis=0) ) )




    ##################################################
    #################   xgboost   ####################
    ##################################################
    
    
    if run_xgboost:
        xgb_performances = []

        if has_test:
            for random_state in range(start_random, end_random):
            
                print('\n{}\nxgboost  random:{}\n{}'.format('-'*50, random_state, '-'*50 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                ''' API:  https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier '''
            
                ## 'objective': 'binary:logistic' / 'multi:softmax'
                if( len(np.unique(y))==2 ):
                    xgb_objective = 'binary:logistic'
                else:
                    xgb_objective = 'multi:softmax'
                print('  xgboost objective: {}  ,  Data shape: X:{} , X_test:{}'.format(xgb_objective, X.shape, X_test.shape) )
                
                param_dist = {'objective': xgb_objective, 'n_estimators':500, 'n_jobs':-1 , 'random_state':random_state }
                clf = XGBClassifier(**param_dist)
                clf.fit(X, y)
                
                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score( y_test, y_pred, average="weighted")

                print('xgboost acc: {}, auc:{}, weighted_f1:{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    xgb_performances.append(acc)
                elif metric_all == 'roc_auc':
                    xgb_performances.append(roc_auc)
                elif metric_all == 'f1':
                    xgb_performances.append(f1)
                else:
                    raise ValueError('metric ERROR')

            pass

        else:
            
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold

            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n xgboost kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                ## 'objective': 'binary:logistic' / 'multi:softmax'
                if( len(np.unique(y))==2 ):
                    xgb_objective = 'binary:logistic'
                else:
                    xgb_objective = 'multi:softmax'
                print('  xgboost objective: {}  ,  Data shape: X:{} , X_test:{}'.format(xgb_objective, X.shape, X_test.shape) )
                
                param_dist = {'objective': xgb_objective, 'n_estimators':500, 'n_jobs':-1 , 'random_state':random_state }
                clf = XGBClassifier(**param_dist)
                clf.fit(X, y)
                
                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]

                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score( y_test, y_pred, average="weighted")

                print('xgboost acc: {}, auc:{}, weighted_f1:{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    xgb_performances.append(acc)
                elif metric_all == 'roc_auc':
                    xgb_performances.append(roc_auc)
                elif metric_all == 'f1':
                    xgb_performances.append(f1)
                else:
                    raise ValueError('metric ERROR')

            pass

        print('\nxgboost test acc：\n {}\n'.format( np.mean(xgb_performances,axis=0) ) )



    
    ####################################################
    ###### sklearn GradientBoostingClassifier ##########
    ####################################################
    
    
    if run_gradientboosting:
        performances = []

        if has_test:
            for random_state in range(start_random, end_random):
            
                print('\n{}\n sklearn GradientBoostingClassifier random:{}\n{}'.format('#'*60, random_state, '#'*60 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                print('\nRun sklearn GradientBoostingClassifier...\n' )
                
                clf = GradientBoostingClassifier( n_estimators=500, random_state=random_state , max_depth=10 )
                clf.fit( X, y )

                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]

                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score( y_test, y_pred, average="weighted")

                print('GradientBoostingClassifier acc: {}, auc:{}, f1{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    performances.append(acc)
                elif metric_all == 'roc_auc':
                    performances.append(roc_auc)
                elif metric_all == 'f1':
                    performances.append(f1)
                else:
                    raise ValueError('metric ERROR')


        else:
            
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold
            
            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n sklearn GradientBoostingClassifier kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                print('\nRun sklearn GradientBoostingClassifier...\n' )
                
                clf = GradientBoostingClassifier( n_estimators=500, random_state=random_state , max_depth=10 )
                clf.fit( X, y )
                
                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                
                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score( y_test, y_pred, average="weighted")

                print('GradientBoostingClassifier acc: {}, auc:{}, f1{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    performances.append(acc)
                elif metric_all == 'roc_auc':
                    performances.append(roc_auc)
                elif metric_all == 'f1':
                    performances.append(f1)
                else:
                    raise ValueError('metric ERROR')


        print('\nsklearn GradientBoostingClassifier  test acc ：\n {}\n'.format( np.mean(performances,axis=0) ) )


    
    ####################################################
    ########## sklearn Random Forest ###################
    ####################################################
    
    if run_RandomForest:
        performances = []

        if has_test:
            ##  
            for random_state in range(start_random, end_random):
            
                print('\n{}\n sklearn RandomForestClassifier random:{}\n{}'.format('#'*60, random_state, '#'*60 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                print('\nRun sklearn RandomForestClassifier...\n' )
                
                clf = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1) 
                clf.fit( X, y )
                
                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                
                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score(y_test, y_pred, average='weighted')

                print('RandomForestClassifier acc: {}, auc:{}, f1:{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    performances.append(acc)
                elif metric_all == 'roc_auc':
                    performances.append(roc_auc)
                elif metric_all == 'f1':
                    performances.append(f1)
                else:
                    raise ValueError('metric ERROR')

        else:
            ##  
            
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold
            
            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n sklearn RandomForestClassifier kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                print('\nRun sklearn RandomForestClassifier...\n')
                clf = RandomForestClassifier( n_estimators=500, random_state=random_state, n_jobs=-1 )
                clf.fit( X, y )
                
                y_pred = clf.predict( X_test )
                y_pred_proba = clf.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                
                acc = accuracy_score(y_test, y_pred) 
                roc_auc = roc_auc_score( y_test, y_pred_proba, average="weighted", multi_class="ovr" )
                f1 = f1_score(y_test, y_pred, average='weighted')

                print('RandomForestClassifier acc: {}, auc:{}, f1:{}\n'.format( acc , roc_auc, f1 ) )
                
                if metric_all== 'acc':
                    performances.append(acc)
                elif metric_all == 'roc_auc':
                    performances.append(roc_auc)
                elif metric_all == 'f1':
                    performances.append(f1)
                else:
                    raise ValueError('metric ERROR')
                


        print('\nRandomForestClassifier  test acc ：\n {}\n'.format( np.mean(performances,axis=0) ) )



    

    ####################################################
    ############## sklearn Gaussian SVM ################
    ####################################################
    
    if run_RBF_SVC:
        performances = []
        f1_performances = []
        
        if has_test:
             for random_state in range(start_random, end_random):
            
                print('\n{}\n sklearn Gaussian SVM random:{}\n{}'.format('#'*60, random_state, '#'*60 ) )
                print('{}'.format( datetime.datetime.now()) )

                X, y = X_raw, y_raw

                X, y = resample_train( IF_resample_train, X, y, resample_num_train, random_state)

                print('\nRun sklearn Gaussian SVM...\n' )
                
                clf = SVC( kernel='rbf', random_state= random_state)
                clf.fit( X, y )
                
                y_pred = clf.predict( X_test )
                
                acc = accuracy_score(y_test, y_pred) 
                f1 = f1_score(y_test, y_pred, average='weighted')

                print('Gaussian SVM acc: {}, f1:{}\n'.format( acc , f1 ) )
                
                performances.append(acc)
                f1_performances.append(f1)

                if metric_all != 'acc' and metric_all != 'f1' :
                    raise ValueError('metric ERROR')

        else:
             
            X_raw1, y_raw1 = resample_data( IF_resample_all, X_raw, y_raw, resample_num_all, random_state )

            kf = StratifiedKFold( 5, shuffle=True, random_state=42 )  ##  KFold / StratifiedKFold
            
            i = 0
            for train_index, test_index in kf.split(X_raw1, y_raw1):
                print('\n{}\n sklearn Gaussian SVM kfold:{}\n{}'.format('#'*60, i, '#'*60 ) )
                i+=1
                print('{}'.format( datetime.datetime.now()) )

                X = X_raw1[train_index, :]
                X_test = X_raw1[test_index, :]
                y = y_raw1[train_index]
                y_test = y_raw1[test_index]

                print('\nRun sklearn Gaussian SVM...\n')
                clf = SVC( kernel='rbf', random_state= random_state)
                clf.fit( X, y )
                
                y_pred = clf.predict( X_test )
                
                acc = accuracy_score(y_test, y_pred  ) 
                f1 = f1_score(y_test, y_pred, average='weighted')

                print('Gaussian SVM acc: {}, f1:{}\n'.format( acc , f1 ) )
                
                performances.append(acc)
                f1_performances.append(f1)
                
                if metric_all != 'acc' and metric_all != 'f1' :
                    raise ValueError('metric ERROR')
                


        print('\nGaussian SVM  ： {} , std:{}, \ntest f1 ： {} , std:{},\n'.format( \
                np.mean(performances,axis=0) , np.std(performances,axis=0), np.mean(f1_performances,axis=0), np.std(f1_performances,axis=0) ) )



        

    

import time
import datetime

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from hiDF.gcForest.layer import *







if __name__ == "__main__":
    

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    print('{}\n{}\n{}\n'.format( '*'*50, '*'*50, datetime.datetime.now()) )

    hiDF_test()
    
    print('\n{}\n{}\n{}'.format(datetime.datetime.now(), '*'*50, '*'*50 ) )
    
    


