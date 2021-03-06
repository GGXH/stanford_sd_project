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
    variable_list = ['ADL', 'ADX', 'ATR', 'Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume', 'Aroon_down', 'Aroon_osc', 'Aroon_up', 'BBB', 'BP', 'CGI', 'CMF_20', 'Chain_osc', 'Close', 'Close_10_1', 'Close_10_10', 'Close_10_11', 'Close_10_12', 'Close_10_13', 'Close_10_14', 'Close_10_15', 'Close_10_16', 'Close_10_17', 'Close_10_18', 'Close_10_19', 'Close_10_2', 'Close_10_20', 'Close_10_21', 'Close_10_22', 'Close_10_23', 'Close_10_24', 'Close_10_25', 'Close_10_26', 'Close_10_27', 'Close_10_28', 'Close_10_29', 'Close_10_3', 'Close_10_30', 'Close_10_4', 'Close_10_5', 'Close_10_6', 'Close_10_7', 'Close_10_8', 'Close_10_9', 'Close_1_1', 'Close_1_10', 'Close_1_11', 'Close_1_12', 'Close_1_13', 'Close_1_14', 'Close_1_15', 'Close_1_16', 'Close_1_17', 'Close_1_18', 'Close_1_19', 'Close_1_2', 'Close_1_20', 'Close_1_21', 'Close_1_22', 'Close_1_23', 'Close_1_24', 'Close_1_25', 'Close_1_26', 'Close_1_27', 'Close_1_28', 'Close_1_29', 'Close_1_3', 'Close_1_30', 'Close_1_4', 'Close_1_5', 'Close_1_6', 'Close_1_7', 'Close_1_8', 'Close_1_9', 'Close_20_1', 'Close_20_10', 'Close_20_11', 'Close_20_12', 'Close_20_13', 'Close_20_14', 'Close_20_15', 'Close_20_16', 'Close_20_17', 'Close_20_18', 'Close_20_19', 'Close_20_2', 'Close_20_20', 'Close_20_21', 'Close_20_22', 'Close_20_23', 'Close_20_24', 'Close_20_25', 'Close_20_26', 'Close_20_27', 'Close_20_28', 'Close_20_29', 'Close_20_3', 'Close_20_30', 'Close_20_4', 'Close_20_5', 'Close_20_6', 'Close_20_7', 'Close_20_8', 'Close_20_9', 'Close_5_1', 'Close_5_10', 'Close_5_11', 'Close_5_12', 'Close_5_13', 'Close_5_14', 'Close_5_15', 'Close_5_16', 'Close_5_17', 'Close_5_18', 'Close_5_19', 'Close_5_2', 'Close_5_20', 'Close_5_21', 'Close_5_22', 'Close_5_23', 'Close_5_24', 'Close_5_25', 'Close_5_26', 'Close_5_27', 'Close_5_28', 'Close_5_29', 'Close_5_3', 'Close_5_30', 'Close_5_4', 'Close_5_5', 'Close_5_6', 'Close_5_7', 'Close_5_8', 'Close_5_9', 'CopCv', 'DI14N', 'DI14P', 'DMN', 'DMN_14WSM', 'DMP', 'DMP_14WSM', 'DPO', 'DX', 'EMV', 'Ex-Dividend', 'FI1', 'FI13', 'High', 'KST', 'KST_sng_line', 'LBB', 'Low', 'MACD_hist', 'MACD_line', 'MBB', 'MFI', 'MFM', 'MFV', 'MI', 'NVI', 'OBV', 'Open', 'PBI', 'PMO_line', 'PMO_sign_line', 'PPO', 'PPO_hist', 'PVI', 'PVO_hist', 'PVO_line', 'RCMA1', 'RCMA2', 'RCMA3', 'RCMA4', 'ROC_10', 'ROC_100', 'ROC_11', 'ROC_14', 'ROC_15', 'ROC_195', 'ROC_20', 'ROC_265', 'ROC_30', 'ROC_390', 'ROC_40', 'ROC_530', 'ROC_65', 'ROC_75', 'RS', 'RSI', 'SK1', 'SK10', 'SK11', 'SK12', 'SK2', 'SK3', 'SK4', 'SK5', 'SK6', 'SK7', 'SK8', 'SK9', 'Slope_10_1', 'Slope_10_2', 'Slope_10_3', 'Slope_10_4', 'Slope_1_1', 'Slope_1_2', 'Slope_1_3', 'Slope_1_4', 'Slope_20_1', 'Slope_20_2', 'Slope_20_3', 'Slope_20_4', 'Slope_5_1', 'Slope_5_2', 'Slope_5_3', 'Slope_5_4', 'SpecK', 'Split Ratio', 'TP', 'TR', 'TRIX', 'TR_14WSM', 'TSI', 'UBB', 'UO', 'UlcerI', 'Volume', 'perc_D', 'perc_K', 'perc_R', 'stochRSI']    
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        best_file = open(sys.argv[2],'r')
        data_x = np.load(sys.argv[3])
        data_y = np.load(sys.argv[4])
        data_no = data_x.shape[0]
        print data_x.shape
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
            x_list = sorted([variable_list.index(key) for key in info])
            #print info
            print x_list
            #sys.exit(0)
            train_x = data_x[range(0,int(0.7*data_no)), x_list]
            train_y = data_y[range(0,int(0.7*data_no))]
            val_x = data_x[range(int(0.7*data_no)+1, data_no), x_list]
            val_y = data_y[range(int(0.7*data_no)+1, data_no)]
            if len(train_x.shape) == 1:
                train_x = train_x.reshape(-1,1)
                val_x = val_x.reshape(-1,1)
            print train_x.shape
            #sys.exit(0)
            #print train_y.shape
            #sys.exit(0)
            #train_x, train_y, val_x, val_y, test_x, test_y = Get_data(coll, info, 'movement1')
            ##---
            model_file_name = sys.argv[1] + '_logistical_' + str(len(info)) + '.pkl'
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
            conf_mat_file_train.flush()
            conf_mat_file_val.flush()
        conf_mat_file_train.close()
        conf_mat_file_val.close()
