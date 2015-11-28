import sys
from pymongo import MongoClient
import numpy as np
from sklearn import linear_model
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
    name_list_unwanted = ['_id', 'data_id', 'Date', 'price_diff', 'movement1']
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
    mongoclient = MongoClient()
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        best_file = open(sys.argv[2],'r')
        conf_mat_file_train = open(sys.argv[1]+'_forward_confu_mat_train.txt', 'w')
        conf_mat_file_val = open(sys.argv[1]+'_forward_confu_mat_val.txt', 'w')
        train_y = np.empty(2)
        train_y.fill(1)
        train_conf_mat =  Get_confusion_matrix(train_y, train_y)
        line = 'No'
        key_list = sorted(train_conf_mat.keys())
        for key in key_list:
            line += ', ' + key
        conf_mat_file_train.write(line)
        conf_mat_file_val.write(line)
        for line in best_file:
            info = line.split(']')
            info = info[0].replace('[','')
            info = info.replace('\'','')
            info = info.split(',')
            for i in range(0,len(info)):
                info[i] = info[i].lstrip()
                info[i] = info[i].rstrip()
            train_x, train_y, val_x, val_y, test_x, test_y = Get_data(coll, info, 'movement1')
            ##---
            model_file_name = sys.argv[1] + '_RF_' + str(len(info)+1) + '.pkl'
            clf = joblib.load(model_file_name)
            train_y_pred = clf.predict(train_x)
            val_y_pred = clf.predict(val_x)
            train_conf_mat =  Get_confusion_matrix(train_y, train_y_pred)
            val_conf_mat = Get_confusion_matrix(val_y, val_y_pred)
            line_train = str(len(info))
            line_val = str(len(info))
            for key in key_list:
                line_train += ', ' + str(train_conf_mat[key])
                line_val += ', ' + str(val_conf_mat[key])
            line_train += '\n'
            line_val += '\n'
            conf_mat_file_train.write(line_train)
            conf_mat_file_val.write(line_val)
        conf_mat_file_train.close()
        conf_mat_file_val.close()
