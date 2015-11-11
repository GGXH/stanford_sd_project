import sys
from pymongo import MongoClient
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import copy

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


if __name__ == '__main__':
    variable_list = ['ADL', 'ADX', 'ATR', 'Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume', 'Aroon_down', 'Aroon_osc', 'Aroon_up', 'BBB', 'BP', 'CGI', 'CMF_20', 'Chain_osc', 'Close', 'Close_10_1', 'Close_10_10', 'Close_10_11', 'Close_10_12', 'Close_10_13', 'Close_10_14', 'Close_10_15', 'Close_10_16', 'Close_10_17', 'Close_10_18', 'Close_10_19', 'Close_10_2', 'Close_10_20', 'Close_10_21', 'Close_10_22', 'Close_10_23', 'Close_10_24', 'Close_10_25', 'Close_10_26', 'Close_10_27', 'Close_10_28', 'Close_10_29', 'Close_10_3', 'Close_10_30', 'Close_10_4', 'Close_10_5', 'Close_10_6', 'Close_10_7', 'Close_10_8', 'Close_10_9', 'Close_1_1', 'Close_1_10', 'Close_1_11', 'Close_1_12', 'Close_1_13', 'Close_1_14', 'Close_1_15', 'Close_1_16', 'Close_1_17', 'Close_1_18', 'Close_1_19', 'Close_1_2', 'Close_1_20', 'Close_1_21', 'Close_1_22', 'Close_1_23', 'Close_1_24', 'Close_1_25', 'Close_1_26', 'Close_1_27', 'Close_1_28', 'Close_1_29', 'Close_1_3', 'Close_1_30', 'Close_1_4', 'Close_1_5', 'Close_1_6', 'Close_1_7', 'Close_1_8', 'Close_1_9', 'Close_20_1', 'Close_20_10', 'Close_20_11', 'Close_20_12', 'Close_20_13', 'Close_20_14', 'Close_20_15', 'Close_20_16', 'Close_20_17', 'Close_20_18', 'Close_20_19', 'Close_20_2', 'Close_20_20', 'Close_20_21', 'Close_20_22', 'Close_20_23', 'Close_20_24', 'Close_20_25', 'Close_20_26', 'Close_20_27', 'Close_20_28', 'Close_20_29', 'Close_20_3', 'Close_20_30', 'Close_20_4', 'Close_20_5', 'Close_20_6', 'Close_20_7', 'Close_20_8', 'Close_20_9', 'Close_5_1', 'Close_5_10', 'Close_5_11', 'Close_5_12', 'Close_5_13', 'Close_5_14', 'Close_5_15', 'Close_5_16', 'Close_5_17', 'Close_5_18', 'Close_5_19', 'Close_5_2', 'Close_5_20', 'Close_5_21', 'Close_5_22', 'Close_5_23', 'Close_5_24', 'Close_5_25', 'Close_5_26', 'Close_5_27', 'Close_5_28', 'Close_5_29', 'Close_5_3', 'Close_5_30', 'Close_5_4', 'Close_5_5', 'Close_5_6', 'Close_5_7', 'Close_5_8', 'Close_5_9', 'CopCv', 'DI14N', 'DI14P', 'DMN', 'DMN_14WSM', 'DMP', 'DMP_14WSM', 'DPO', 'DX', 'EMV', 'Ex-Dividend', 'FI1', 'FI13', 'High', 'KST', 'KST_sng_line', 'LBB', 'Low', 'MACD_hist', 'MACD_line', 'MBB', 'MFI', 'MFM', 'MFV', 'MI', 'NVI', 'OBV', 'Open', 'PBI', 'PMO_line', 'PMO_sign_line', 'PPO', 'PPO_hist', 'PVI', 'PVO_hist', 'PVO_line', 'RCMA1', 'RCMA2', 'RCMA3', 'RCMA4', 'ROC_10', 'ROC_100', 'ROC_11', 'ROC_14', 'ROC_15', 'ROC_195', 'ROC_20', 'ROC_265', 'ROC_30', 'ROC_390', 'ROC_40', 'ROC_530', 'ROC_65', 'ROC_75', 'RS', 'RSI', 'SK1', 'SK10', 'SK11', 'SK12', 'SK2', 'SK3', 'SK4', 'SK5', 'SK6', 'SK7', 'SK8', 'SK9', 'Slope_10_1', 'Slope_10_2', 'Slope_10_3', 'Slope_10_4', 'Slope_1_1', 'Slope_1_2', 'Slope_1_3', 'Slope_1_4', 'Slope_20_1', 'Slope_20_2', 'Slope_20_3', 'Slope_20_4', 'Slope_5_1', 'Slope_5_2', 'Slope_5_3', 'Slope_5_4', 'SpecK', 'Split Ratio', 'TP', 'TR', 'TRIX', 'TR_14WSM', 'TSI', 'UBB', 'UO', 'UlcerI', 'Volume', 'perc_D', 'perc_K', 'perc_R', 'stochRSI']
    #variable_list = ['ADL', 'ADX', 'ATR', 'Adj Close', 'Adj High', 'Adj Low', 'Adj Open', 'Adj Volume', 'Aroon_down']
    mongoclient = MongoClient()
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        train_x, train_y, val_x, val_y, test_x, test_y = Get_data(coll, variable_list, 'movement1')
        ##---
        best_variable_i = []
        best_variable_l = []
        all_output = open(sys.argv[1]+'all_output_logistical_regression.txt', 'wb')
        best_output = open(sys.argv[1]+'best_output_logistical_regression.txt', 'wb')
        variable_list_best = copy.deepcopy(variable_list)
        variable_i_best = range(0, len(variable_list_best))
        while len(variable_list_best) > 1:
            best_error = 20000000
            best_i = -1
            for i in variable_i_best:
                try_variable_i = copy.deepcopy(variable_i_best)
                try_variable_l = copy.deepcopy(variable_list_best)
                try_variable_l.pop(try_variable_i.index(i))
                try_variable_i.remove(i)
                try_train_x = train_x[:, try_variable_i]
                try_val_x = val_x[:,try_variable_i]
                clf = svm.SVC()
                clf.fit(try_train_x, train_y)
                val_y_pred = clf.predict(try_val_x)
                train_y_pred = clf.predict(try_train_x)
                diff_val_y = sum(abs(val_y_pred - val_y)/2)
                diff_train_y = sum(abs(train_y_pred - train_y)/2)               
                print>>all_output, try_variable_l, try_variable_i, diff_train_y, diff_val_y
                if diff_val_y < best_error:
                    best_error = diff_val_y
                    best_i = i
                    clf_best = clf
            print best_i
            file_name = sys.argv[1]+'_svm_svc_'+str(len(variable_list_best))+'.pkl'
            joblib.dump(clf_best, file_name, compress=9)
            variable_list_best.pop(variable_i_best.index(best_i))
            variable_i_best.remove(best_i)
            print>>best_output, variable_list_best, best_error
            print variable_list[best_i]
