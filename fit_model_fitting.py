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
    confusion_matrix = {'FN': 0, 'FP': 0, 'TN': 0, 'TP': 0}
    total = len(real_y)
    for i in range(0, total):
        if real_y[i] < 0:
            if pred_y[i] < 0:
                confusion_matrix['TN'] += 1
            else:
                confusion_matrix['FP'] += 1
        else:
            if pred_y[i] < 0:
                confusion_matrix['FN'] += 1
            else:
                confusion_matrix['TP'] += 1
    print confusion_matrix
    confusion_matrix['accuracy'] = ( confusion_matrix['TP'] + confusion_matrix['TN'] ) * 1. / total
    confusion_matrix['recall'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FN'] )
    confusion_matrix['F1'] = confusion_matrix['TP'] * 2. / ( 2. * confusion_matrix['TP'] + confusion_matrix['FN'] + confusion_matrix['FP'] )
    tmp = confusion_matrix['TP'] * confusion_matrix['TN'] - confusion_matrix['FP'] * confusion_matrix['FN']
    if tmp != 0:
        confusion_matrix['MCC'] = tmp * 1. / ( (  confusion_matrix['TP'] +  confusion_matrix['FP'] ) * (  confusion_matrix['TP'] +  confusion_matrix['FN'] ) * (  confusion_matrix['TN'] +  confusion_matrix['FP'] ) * (  confusion_matrix['TN'] +  confusion_matrix['FN'] ) )**0.5
    else:
        confusion_matrix['MCC'] = 0
    if confusion_matrix['TP'] == 0 and confusion_matrix['FP'] == 0:
        confusion_matrix['precision'] = 0
    else:
        confusion_matrix['precision'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FP'] )
    return confusion_matrix



def Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y):
    '''fit a model and output the confusion matrix '''
    clf.fit(train_x, train_y)
    model_name = '_'.join([str(x) for x in model_param])
    file_name = file_name_pre + '_' + model_name + '.pkl'
    joblib.dump(clf, file_name, compress=9)
    val_y_pred = clf.predict(val_x)
    train_y_pred = clf.predict(train_x)
    val_confusion = Get_confusion_matrix(val_y, val_y_pred)
    train_confusion = Get_confusion_matrix(train_y, train_y_pred)
    name_list = file_name.split('.')
    print>>f, ''
    print>>f, model_param
    print>>f, 'val confusion matrix:'
    print>>f, val_confusion
    print>>f, 'train confusion matrix'
    print>>f, train_confusion
    return val_confusion



if __name__ == '__main__':
    ##--load data
    data_x = np.load(sys.argv[3])
    data_y = np.load(sys.argv[4])    
    print data_x.shape
    data_no = data_x.shape[0]
    train_x = data_x[range(0,int(0.7*data_no)), :]
    train_y = data_y[range(0,int(0.7*data_no))]
    val_x = data_x[range(int(0.7*data_no)+1, data_no), :]
    val_y = data_y[range(int(0.7*data_no)+1, data_no)] 
    ##---
    file_name_pre = sys.argv[1] + '_' + sys.argv[2]
    f = open(file_name_pre+'_confusion.txt', 'wb')
    if sys.argv[2] == 'lc':
        for pen in ['l1', 'l2']:
            for c_reg in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                clf = linear_model.LogisticRegression(penalty=pen, C=c_reg)
                model_param = [pen, c_reg]
                Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
    elif sys.argv[2] == 'svm':
        for c_reg in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            for kern_fun in ['poly']:# ['linear', 'poly', 'rbf', 'sigmoid']:
                if kern_fun in ['rbf', 'poly', 'sigmoid']:
                    for gam_coef in ['auto', 0.01, 0.1, 0.5, 0.8]: #['auto', 0.001, 0.01, 0.1, 0.5, 0.8]:
                        if kern_fun in ['poly', 'sigmoid']:
                            for coef0_val in [10]: #[0.001, 0.01, 0.1, 1, 10]:
                                if kern_fun == 'poly':
                                    for deg_coef in [8, 10]: #[1, 2, 3, 5, 8, 10]:
                                        clf = svm.SVC(C=c_reg, kernel=kern_fun, gamma=gam_coef, coef0=coef0_val, degree=deg_coef)
                                        model_param = [c_reg, kern_fun, gam_coef, coef0_val, deg_coef]
                                        val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
                                else:
                                    clf = svm.SVC(C=c_reg, kernel=kern_fun, gamma=gam_coef, coef0=coef0_val)
                                    model_param = [c_reg, kern_fun, gam_coef, coef0_val]
                                    val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
                        else:
                            clf = svm.SVC(C=c_reg, kernel=kern_fun, gamma=gam_coef)
                            model_param = [c_reg, kern_fun, gam_coef]
                            val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
                else:
                    clf = svm.SVC(C=c_reg, kernel=kern_fun)
                    model_param = [c_reg, kern_fun]
                    val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
    elif sys.argv[2] == 'RF':
        for n in [2, 3, 5, 8, 10, 15]:
            for crit in ['gini', 'entropy']:
                for max_feat in ['auto', 'log2', None, 0.1, 0.2, 0.4, 0.8]:
                    clf = RandomForestClassifier(n_estimators = n, criterion = crit, max_features = max_feat)
                    model_param = [n, crit, max_feat]
                    val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
