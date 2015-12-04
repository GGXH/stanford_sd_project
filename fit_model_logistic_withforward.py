import sys
from pymongo import MongoClient
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import copy

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


def Get_best_data(best_file, variable_list, train_x, val_x):
    '''get data from file'''
    f = open(best_file,'r')
    max_parameter = []
    max_val_mcc = 0
    param = eval(f.readline().rstrip())
    end = 0
    while end == 0:
        dummy = f.readline()
        test_conf = eval(f.readline().rstrip())
        dummy = f.readline()
        train_conf = eval(f.readline().rstrip())
        if test_conf['MCC'] > max_val_mcc:
            max_parameter = copy.deepcopy(param)
            max_val_mcc = test_conf['MCC']
        #print len(param)
        if len(param) == len(variable_list) - 1:
            end = 1
        else:
            param = eval(f.readline().rstrip())
    print max_parameter
    print len(max_parameter)
    print max_val_mcc
    ###--
    item_no = []
    for item in max_parameter:
        item_no.append(variable_list.index(item))
    item_no = sorted(item_no)
    return train_x[:,item_no], val_x[:,item_no]



if __name__ == '__main__':
    variable_list = ['Slope_half_20_2', 'Slope_half_20_3', 'Slope_half_20_4', 'Close_1_12', 'RCMA1', 'RCMA2', 'RCMA3', 'BP', 'TP', 'Close_10_19', 'Close_10_18', 'Close_10_17', 'Close_10_16', 'Close_10_15', 'Close_10_14', 'Close_10_13', 'Close_10_12', 'Close_10_11', 'Close_10_10', 'Close_1_30', 'FI1', 'Close_20_18', 'Close_20_19', 'RS', 'Close_20_14', 'Close_20_15', 'Close_20_16', 'Close_20_17', 'Close_20_10', 'Close_20_11', 'Close_20_12', 'Close_20_13', 'Adj High', 'LBB', 'ROC_75', 'SK11', 'Aroon_up', 'PPO_hist', 'SK12', 'SpecK', 'Close_20_20', 'Close_20_23', 'KST_sng_line', 'Close_20_22', 'PVO_hist', 'MACD_hist', 'FI13', 'Close_20_25', 'Close_20_24', 'Close_1_27', 'Close_1_26', 'Close_1_25', 'Close_1_24', 'Close_1_23', 'Close_1_22', 'Close_1_21', 'Close_1_20', 'Close_20_29', 'Close_20_28', 'Pred_Close_1_1', 'Close_20_26', 'Close_1_29', 'Pred_Close_1_4', 'Pred_Close_1_3', 'ROC_390', 'Pred_Close_1_2', 'RCMA4', 'ROC_65', 'Aroon_down', 'KST', 'DMN_14WSM', 'TSI', 'DMN', 'DMP', 'Close_10_30', 'EMV', 'Close_1_5', 'Chain_osc', 'Slope_half_10_4', 'Close_10_24', 'Close_1_28', 'DMP_14WSM', 'Close_20_30', 'MFI', 'Close_1_4', 'MFM', 'Close', 'MFV', 'Slope_10_1', 'ROC_10', 'ROC_11', 'ROC_14', 'ROC_15', 'PMO_sign_line', 'Slope_10_4', 'PMO_line', 'Slope_1_4', 'Slope_10_2', 'Slope_10_3', 'Slope_1_1', 'Slope_1_3', 'Slope_1_2', 'Close_5_12', 'Close_5_13', 'Close_1_6', 'Close_5_11', 'Close_10_22', 'Close_10_23', 'Close_5_14', 'Close_1_3', 'Close_5_18', 'ROC_265', 'Close_1_8', 'Close_1_9', 'Close_10_28', 'Close_10_29', 'SK5', 'Close_10_26', 'SK7', 'SK6', 'SK1', 'SK3',  'Close_5_10', 'Slope_5_3', 'Slope_5_2', 'SK9', 'SK8', 'Close_5_1', 'Close_5_2', 'Close_10_25', 'Close_5_4', 'Close_5_5', 'Close_5_6', 'Close_5_7', 'Close_5_8', 'Close_5_16', 'Close_20_27', 'NVI', 'Close_1_1', 'Close_10_7', 'Close_10_6', 'Close_10_5', 'ROC_100', 'Close_10_3', 'Close_10_20', 'Close_1_7', 'Close_10_21', 'Slope_20_2', 'perc_R', 'Close_10_9', 'Close_10_8', 'Close_10_27', 'perc_K', 'Low', 'perc_D', 'UlcerI', 'CGI', 'DI14N', 'Close_5_19', 'ROC_30', 'PBI', 'Adj Low', 'PVO_line', 'Close_1_10', 'Close_5_22', 'DI14P', 'Aroon_osc', 'Close_5_17', 'SK4', 'ATR', 'ADX', 'PVI', 'Close_5_30', 'TRIX', 'OBV', 'Close_10_4', 'ADL', 'CMF_20', 'Close_1_2', 'Slope_half_10_3', 'ROC_20', 'Slope_half_10_2', 'Slope_half_1_4', 'Slope_20_4', 'SK2', 'Slope_20_1', 'ROC_195', 'Slope_20_3', 'Slope_half_1_3', 'Slope_5_1', 'Close_20_21', 'Split Ratio', 'Open', 'Close_5_15', 'Close_5_29', 'Close_5_28', 'TR_14WSM', 'SK10', 'Close_5_23', 'MI', 'Close_5_21', 'Close_5_20', 'Close_5_27', 'Close_5_26', 'Close_5_25', 'Close_5_24', 'Close_20_6', 'Close_20_7', 'Close_20_4', 'Close_20_5', 'Close_10_2', 'Close_20_3', 'DX', 'Close_20_1', 'DPO', 'Close_20_8', 'Close_20_9', 'Close_1_18', 'Close_1_19', 'Close_1_16', 'Close_1_17', 'Close_1_14', 'Close_1_15', 'PPO', 'Close_1_13', 'MBB', 'Close_1_11', 'Volume', 'Close_10_1', 'Close_5_3', 'Close_20_2', 'CopCv', 'Adj Close', 'UO', 'BBB', 'Adj Volume', 'Slope_half_5_2', 'Slope_half_5_3', 'Slope_half_5_4', 'Slope_half_1_2', 'Close_5_9', 'Adj Open', 'RSI', 'Ex-Dividend', 'TR', 'ROC_530', 'UBB', 'High', 'Slope_5_4', 'ROC_40', 'MACD_line', 'stochRSI']
    ##--load data
    data_x = np.load(sys.argv[3])
    data_y = np.load(sys.argv[4])    
    print data_x.shape
    data_no = data_x.shape[0]
    train_x = data_x[range(0,int(0.7*data_no)), :]
    train_y = data_y[range(0,int(0.7*data_no))]
    val_x = data_x[range(int(0.7*data_no)+1, data_no), :]
    val_y = data_y[range(int(0.7*data_no)+1, data_no)] 
    ##--
    train_x, val_x = Get_best_data(sys.argv[5], variable_list, train_x, val_x)
    print train_x.shape
    print val_x.shape
    #sys.exit(0)
    ##---
    file_name_pre = sys.argv[1] + '_' + sys.argv[2]
    f = open(file_name_pre+'_confusion.txt', 'wb')
    if sys.argv[2] == 'lc':
        for pen in ['l2']: #['l1', 'l2']:
            for c_reg in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9]: #[0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                clf = linear_model.LogisticRegression(penalty=pen, C=c_reg)
                model_param = [pen, c_reg]
                Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
    elif sys.argv[2] == 'svm':
        for c_reg in [0.001, 0.01, 0.1, 1]:  #[0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            for kern_fun in ['poly']: #['linear', 'rbf', 'sigmoid']: #['linear', 'poly', 'rbf', 'sigmoid']:
                if kern_fun in ['rbf', 'poly', 'sigmoid']:
                    for gam_coef in ['auto', 0.001, 0.01, 0.1, 0.5, 0.8]:
                        if kern_fun in ['poly', 'sigmoid']:
                            for coef0_val in [1, 10]: #[0.001, 0.01, 0.1, 1, 10]:
                                if kern_fun == 'poly':
                                    for deg_coef in [5, 8]: #[1, 2, 3, 5, 8, 10]:
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
        for n in [6, 7, 8, 9]: #[2, 3, 5, 8, 10, 15]:
            for crit in ['gini', 'entropy']:
                for max_feat in ['auto']: #['auto', 'log2', None, 0.1, 0.2, 0.4, 0.8]:
                    clf = RandomForestClassifier(n_estimators = n, criterion = crit, max_features = max_feat)
                    model_param = [n, crit, max_feat]
                    val_conf = Fit_model_output_result(clf, f, file_name_pre, model_param, train_x, train_y, val_x, val_y)
