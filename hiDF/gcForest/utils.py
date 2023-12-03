import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_accuracy(label, predict):
    if len(predict.shape) > 1:
        test = np.argmax(predict, axis=1)
    else:
        test = predict
    
    #print( label[0:3], test[0:3] )
    #label = np.argmax(label, axis=1)
    test_copy = test.astype("int")
    label_copy = label.astype("int").ravel()
    #print( test_copy.shape , label_copy.shape , np.sum(test_copy == label_copy) )
    acc = np.sum(test_copy == label_copy) * 1.0 / len(label_copy) * 100
    acc2 = accuracy_score(label_copy, test_copy)
    #print( 'acc:',  acc, acc2 )
    return acc


# normalize each col in X
def data_normal(X, method="z"):
    if method == "minmax":
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(X)
        return X_minMax
    elif method == "z":
        std = preprocessing.StandardScaler()
        X_z = std.fit_transform(X)
        return X_z
    else:
        std = preprocessing.StandardScaler()
        X_z = std.fit_transform(X)
        return X_z


# add noise to X
def add_noise(X, percent):
    return X + percent * np.random.randn(X.shape[0], X.shape[1])


def add_noise_label(y, percent):
    n = len(y)
    num = int(percent * n)
    a = shuffle(range(0, n))
    a = a[0:num]
    num_classes = np.max(y) + 1
    noise = np.random.randint(0, num_classes, size=num)
    noise_label = y.copy()
    noise_label[a] = noise
    return noise_label


def compute_gamma(val_prob, train_label):
    train_probs = val_prob.copy()
    p = np.zeros(len(train_label))
    for i in range(len(train_label)):
        p[i] = val_prob[i, train_label[i]]
    for i in range(len(train_label)):
        train_probs[i, train_label[i]] = -1.0
    p2 = np.max(train_probs, axis=1)
    gamma = p - p2
    return gamma


def save_prob(directory, save_list, len):
    for i in range(len):
        path = directory + "_layer{}.txt".format(str(i))
        np.savetxt(path, save_list[i], fmt="%.6f")


def save_acc(path, save_list, len):
    temp = np.array(save_list)
    temp = temp[0:len]
    np.savetxt(path, temp, fmt="%.3f")
