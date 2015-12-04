import sys
from pymongo import MongoClient
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def Get_data_local(coll, variable_list, y_name, i_start, i_end):
    '''Get data from collection local'''
    variable = []
    y = []
    for i in range(i_start, i_end+1):
        local_variable = []
        doc = coll.find_one({'data_id':i})
        for name in variable_list:
            local_variable.append(doc[name])
        variable.append(local_variable)
        y.append(doc[y_name])
    variable = np.array(variable)
    y = np.array(y)
    return variable, y


def Get_data(coll, variable_list, y_name, val_ratio=0.3, test_ratio=0.0001):
    '''Get data from collection'''
    name_list_unwanted = ['_id', 'data_id', 'Date', 'price_diff', 'movement1', 'price_diff_km_2', 'price_diff_km_3', 'price_diff_km_4', 'price_diff_km_5']
    data_number = coll.count()
    train_data_no = int(data_number * ( 1 - val_ratio - test_ratio ))
    val_data_no = int(data_number * val_ratio)
    test_data_no = data_number - train_data_no - val_data_no
    train_x, train_y = Get_data_local(coll, variable_list, y_name, 1, train_data_no)
    val_x, val_y = Get_data_local(coll, variable_list, y_name, train_data_no+1, train_data_no+val_data_no)
    test_x, test_y = Get_data_local(coll, variable_list, y_name, train_data_no+val_data_no+1, data_number)
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
    ##--get RF model
    rf_model = joblib.load(sys.argv[5])
    variable_importance = rf_model.feature_importances_
    ##--fitting model
    for coef_limit in [0, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]: #[0.0015, 0.003, 0.005, 0.007]:
        train_x_no = [i for i in range(0,len(variable_importance)) if variable_importance[i] > coef_limit]
        #print variable_importance
        #print train_x_no
        #print len(train_x_no)
        #sys.exit(0)
        if sys.argv[2] == 'lc':
            clf = linear_model.LogisticRegression()
            clf.fit(train_x[:, train_x_no], train_y)
            joblib.dump(clf, sys.argv[1]+'_RF_' + str(coef_limit) + '_logistical.pkl', compress=9)
        elif sys.argv[2] == 'svm':
            clf = svm.SVC()
            clf.fit(train_x[:, train_x_no], train_y)
            joblib.dump(clf, sys.argv[1]+'_RF_' + str(coef_limit)  + '_svm_svc.pkl', compress=9)
        elif sys.argv[2] == 'RF':
            clf = RandomForestClassifier()
            clf.fit(train_x[:, train_x_no], train_y)
            joblib.dump(clf, sys.argv[1]+'_RF_' + str(coef_limit) + '_RF.pkl', compress=9)
        val_y_pred = clf.predict(val_x[:, train_x_no])
        train_y_pred = clf.predict(train_x[:, train_x_no])
        confusion_matrix_train = Get_confusion_matrix(train_y, train_y_pred)
        confusion_matrix_val = Get_confusion_matrix(val_y, val_y_pred)
        with open(sys.argv[1] + sys.argv[2]+'_confusion.txt', 'a') as f:
            f.write(str(coef_limit) + '\n')
            f.write('train matrix:\n')
            f.write(str(confusion_matrix_train)+'\n')
            f.write('test matrix:\n')
            f.write(str(confusion_matrix_val)+'\n')
            f.write('\n')
