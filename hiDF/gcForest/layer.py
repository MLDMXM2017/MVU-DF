
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# import sys
# sys.path.append("./Forest/pylib")

# import Globals
# import Data
# import ClassificationTree
# import RandomForest
# import ExtraTrees
import math
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from .utils import compute_accuracy, data_normal


class Layer:
    def __init__(self, n_estimators, num_forests, num_classes, max_depth=100, min_samples_leaf=1, sample_weight=None,\
                 random_state=42, purity_function="gini" , bootstrap=True, parallel=False, num_threads=-1 ):
        
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state 
        self.purity_function = purity_function  
        self.bootstrap = bootstrap 
        self.parallel= parallel
        self.num_threads = num_threads
        self.model = []

        if( not self.parallel ):
            self.num_threads = 1


    def train(self, train_data, train_label, val_data):
        
        val_prob = np.zeros([self.num_forests, val_data.shape[0], self.num_classes])

        for forest_index in range(self.num_forests):
            if forest_index % 2 == 0:
                
                clf = RandomForestClassifier(n_estimators=self.n_estimators,  
                                            max_depth=self.max_depth,
                                            min_samples_leaf=self.min_samples_leaf,
                                            random_state=self.random_state+forest_index ,
                                            criterion=self.purity_function,
                                            bootstrap=self.bootstrap,
                                            n_jobs= self.num_threads  )
                
                clf.fit(train_data, train_label)
                val_prob[forest_index, :] = clf.predict_proba(val_data)

            else:

                clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            min_samples_leaf=self.min_samples_leaf,
                                            random_state=self.random_state+forest_index  ,
                                            criterion=self.purity_function,
                                            bootstrap=self.bootstrap,
                                            n_jobs= self.num_threads  )
                
                clf.fit(train_data, train_label)
                val_prob[forest_index, :] = clf.predict_proba(val_data)

            self.model.append(clf)

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)

        return [val_avg, val_concatenate]


    def predict(self, test_data):
        predict_prob = np.zeros([self.num_forests, test_data.shape[0], self.num_classes])
        for forest_index, clf in enumerate(self.model):
            predict_prob[forest_index, :] = clf.predict_proba(test_data)
        
        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0, 2))
        predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

        return [predict_avg, predict_concatenate]


class KfoldWarpper:
    def __init__(self, num_forests, n_estimators, num_classes, n_fold, kf, layer_index, max_depth=20, min_samples_leaf=1, \
                    sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=False, num_threads=-1 ):

        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state 
        self.purity_function = purity_function  
        self.bootstrap = bootstrap 
        self.parallel= parallel
        self.num_threads = num_threads
        self.model = []

    def train(self, train_data, train_label):
        #self.num_classes = int(np.max(train_label) + 1)
        num_classes = int(np.max(train_label) + 1)
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes:{} not equal to actual num_classes:{}".format( self.num_classes, num_classes) )

        num_samples, num_features = train_data.shape

        val_prob = np.empty([num_samples, self.num_classes])
        val_prob_concatenate = np.empty([num_samples, self.num_forests * self.num_classes])

        for train_index, test_index in self.kf.split(train_data, train_label):
            X_train = train_data[train_index, :]
            X_val = train_data[test_index, :]
            y_train = train_label[train_index]

            layer = Layer(self.n_estimators, self.num_forests, self.num_classes, self.max_depth, self.min_samples_leaf, self.sample_weight,\
                            self.random_state, self.purity_function, self.bootstrap , self.parallel, self.num_threads )
            val_prob[test_index], val_prob_concatenate[test_index, :] = \
                layer.train(X_train, y_train, X_val)
            self.model.append(layer)

        return [val_prob, val_prob_concatenate]


    def predict(self, test_data):
    
        test_prob = np.zeros([test_data.shape[0], self.num_classes])
        test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests * self.num_classes])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = \
                layer.predict(test_data)

            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold

        return [test_prob, test_prob_concatenate]
