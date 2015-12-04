import sys
from pymongo import MongoClient
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import copy


if __name__ == '__main__':
    f = open(sys.argv[1],'r')
    max_parameter = []
    max_val_mcc = 0
    line = f.readline()
    param = f.readline().rstrip()
    print param
    end = 0
    while param:
        dummy = f.readline()
        test_conf = eval(f.readline().rstrip())
        dummy = f.readline()
        train_conf = eval(f.readline().rstrip())
        if test_conf['MCC'] > max_val_mcc:
            max_parameter = copy.deepcopy(param)
            max_val_mcc = test_conf['MCC']
            max_val_conf = copy.deepcopy(test_conf)
        print param
        param = f.readline().rstrip()
        param = f.readline().rstrip()
        print param
    print max_parameter
    print max_val_mcc
    print max_val_conf
    sys.exit(0)
