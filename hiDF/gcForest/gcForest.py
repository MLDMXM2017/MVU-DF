from sklearn.model_selection import KFold, StratifiedKFold
from .layer import *


class gcForest1:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1 ):
        
        
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

        self.model = []

    def train(self,train_data, train_label, X_test, y_test):
         # basis information of dataset
        num_classes = int(np.max(train_label) + 1)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        num_samples, num_features = train_data.shape

        # basis process
        train_data_new = train_data.copy()
        test_data_new = X_test.copy()

        # return value
        val_p = []
        val_acc = []
        best_train_acc = 0.0
        best_test_acc=0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0
        # temp = KFold(n_splits=self.n_fold, shuffle=True)
        # kf = []
        # for i, j in temp.split(range(len(train_label))):
        #     kf.append([i, j])


        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state)  ##  KFold / StratifiedKFold


        while layer_index < self.max_layer:

            print("\n--------------\nlayer {},   X_train shape:{}, X_test shape:{}...\n ".format(str(layer_index), train_data_new.shape, test_data_new.shape) )

            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                layer_index, self.max_depth, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                self.purity_function, self.bootstrap,  self.parallel, self.num_threads  )

            val_prob, val_stack= layer.train(train_data_new, train_label)
            test_prob, test_stack = layer.predict( test_data_new )

            train_data_new = np.concatenate([train_data, val_stack], axis=1)
            test_data_new = np.concatenate([X_test, test_stack], axis=1 )

            temp_val_acc = compute_accuracy(train_label, val_prob)
            temp_test_acc = compute_accuracy( y_test, test_prob )
            print("val  acc:{} \nTest acc: {}".format( str(temp_val_acc), str(temp_test_acc)) )
            

            if best_train_acc >= temp_val_acc:
                bad += 1
            else:
                bad = 0
                best_train_acc = temp_val_acc
                best_test_acc = temp_test_acc
                best_layer_index = layer_index
            if bad >= 3:
                break

            layer_index = layer_index + 1

        
        print( 'best layer index: {}, its\' test acc: {} '.format(best_layer_index, best_test_acc) )

        return best_test_acc



    def predict_proba(self, test_data):
        test_data_new = test_data.copy()
        test_prob = []
        for layer in self.model:
            test_prob, test_stack = layer.predict(test_data_new)
            test_data_new = np.concatenate([test_data, test_stack], axis=1)
            layer_index = layer_index + 1

        return test_prob

    def predict(self, test_data):
        test_data_new = test_data.copy()
        test_prob = []
        for layer in self.model:
            test_prob, test_stack = layer.predict(test_data_new)
            test_data_new = np.concatenate([test_data, test_stack], axis=1)

        return np.argmax(test_prob, axis=1)