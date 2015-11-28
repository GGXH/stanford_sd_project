import sys
from pymongo import MongoClient
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def Get_data_local(coll, y_name, i_start, i_end):
    '''Get data from collection local'''
    all_key_list = coll.find_one().keys()
    key_list = []
    for item in all_key_list:
        if item[:3] == 'svd':
            key_list.append(item)
    variable = []
    y = []
    for i in range(i_start, i_end+1):
        local_variable = []
        doc = coll.find_one({'data_id':i})
        for name in key_list:
            local_variable.append(doc[name])
        variable.append(local_variable)
        y.append(doc[y_name])
    variable = np.array(variable)
    y = np.array(y)
    return variable, y


def Get_data(coll, y_name, val_ratio=0.3, test_ratio=0.0001):
    '''Get data from collection'''
    data_number = coll.count()
    train_data_no = int(data_number * ( 1 - val_ratio - test_ratio ))
    val_data_no = int(data_number * val_ratio)
    test_data_no = data_number - train_data_no - val_data_no
    train_x, train_y = Get_data_local(coll, y_name, 1, train_data_no)
    val_x, val_y = Get_data_local(coll, y_name, train_data_no+1, train_data_no+val_data_no)
    test_x, test_y = Get_data_local(coll, y_name, train_data_no+val_data_no+1, data_number)
    return train_x, train_y, val_x, val_y, test_x, test_y


def Get_confusion_matrix(real_y, pred_y):
    '''Get confusion matrix'''
    y_list = sorted(list(set(real_y)))
    print y_list
    total = len(real_y)
    cluster_no = len(y_list)
    confusion_matrix = np.zeros((cluster_no, cluster_no))
    for i in range(0, total):
        confusion_matrix[y_list.index(real_y[i]), y_list.index(pred_y[i])] += 1
    #confusion_matrix['accuracy'] = ( confusion_matrix['TP'] + confusion_matrix['TN'] ) * 1. / total
    #confusion_matrix['recall'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FN'] )
    #confusion_matrix['F1'] = confusion_matrix['TP'] * 2. / ( 2. * confusion_matrix['TP'] + confusion_matrix['FN'] + confusion_matrix['FP'] )
    #if confusion_matrix['TP'] == 0 and confusion_matrix['FP'] == 0:
    #    confusion_matrix['precision'] = 0
    #else:
    #    confusion_matrix['precision'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FP'] )
    return confusion_matrix



def Fit_model_output_result(clf, f, file_name_pre, feature_no, train_x, train_y, val_x, val_y):
    '''fit a model and output the confusion matrix '''
    clf.fit(train_x[:, range(0,feature_no)], train_y)
    file_name = file_name_pre + '_' + str(feature_no) + '.pkl'
    joblib.dump(clf, file_name, compress=9)
    val_y_pred = clf.predict(val_x[:,range(0,feature_no)])
    train_y_pred = clf.predict(train_x[:,range(0,feature_no)])
    val_confusion = Get_confusion_matrix(val_y, val_y_pred)
    train_confusion = Get_confusion_matrix(train_y, train_y_pred)
    name_list = file_name.split('.')
    print>>f, ''
    print>>f, 'feature_no:', feature_no
    print>>f, 'val confusion matrix:'
    print>>f, val_confusion
    print>>f, 'train confusion matrix'
    print>>f, train_confusion
    return val_confusion



if __name__ == '__main__':
    mongoclient = MongoClient()
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        train_x, train_y, val_x, val_y, test_x, test_y = Get_data(coll, sys.argv[3])
        ##---
        file_name_pre = sys.argv[1] + '_svd_' + sys.argv[2] + '_' + sys.argv[3]
        f = open(file_name_pre+'_confusion.txt', 'wb')
        for feature_no in [5, 10, 20, 50, 100, 193]:
            if sys.argv[2] == 'lc':
                clf = linear_model.LogisticRegression()
                Fit_model_output_result(clf, f, file_name_pre, feature_no, train_x, train_y, val_x, val_y)
            elif sys.argv[2] == 'svm':
                clf = svm.SVC()
                val_conf = Fit_model_output_result(clf, f, file_name_pre, feature_no, train_x, train_y, val_x, val_y)
            elif sys.argv[2] == 'RF':
                clf = RandomForestClassifier()
                val_conf = Fit_model_output_result(clf, f, file_name_pre, feature_no, train_x, train_y, val_x, val_y)





