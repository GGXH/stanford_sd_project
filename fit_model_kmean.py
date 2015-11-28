import sys
from pymongo import MongoClient
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.externals import joblib

def Get_data_local(coll, y_name, i_start, i_end):
    '''Get data from collection local'''
    y = []
    for i in range(i_start, i_end+1):
        doc = coll.find_one({'data_id':i})
        y.append(doc[y_name])
    y = np.array(y)
    return y


def Get_data(coll, y_name):
    '''Get data from collection'''
    data_number = coll.count()
    train_y = Get_data_local(coll, y_name, 1, data_number)
    return train_y


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
    confusion_matrix['accuracy'] = ( confusion_matrix['TP'] + confusion_matrix['TN'] ) * 1. / total
    confusion_matrix['recall'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FN'] )
    confusion_matrix['F1'] = confusion_matrix['TP'] * 2. / ( 2. * confusion_matrix['TP'] + confusion_matrix['FN'] + confusion_matrix['FP'] )
    if confusion_matrix['TP'] == 0 and confusion_matrix['FP'] == 0:
        confusion_matrix['precision'] = 0
    else:
        confusion_matrix['precision'] = confusion_matrix['TP'] * 1. / ( confusion_matrix['TP'] + confusion_matrix['FP'] )
    return confusion_matrix



def Fit_model_output_result(clf, f, file_name_pre, feature_no, model_param, train_x, train_y, val_x, val_y):
    '''fit a model and output the confusion matrix '''
    clf.fit(train_x[:, range(0,feature_no)], train_y)
    model_name = '_'.join([str(x) for x in model_param])
    file_name = file_name_pre + '_' + model_name + '.pkl'
    joblib.dump(clf, file_name, compress=9)
    val_y_pred = clf.predict(val_x[:,range(0,feature_no)])
    train_y_pred = clf.predict(train_x[:,range(0,feature_no)])
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
    mongoclient = MongoClient()
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        #train_y = Get_data(coll, 'price_diff')
        #np.save(sys.argv[1]+'_price_diff', train_y)
        train_y = np.load(sys.argv[1]+'_price_diff.npy')
        train_y = train_y.reshape(-1, 1)
        print train_y
        #sys.exit(0)
        ##---
        for n in [2, 3, 4, 5]:
            if sys.argv[2] == 'kmeans':
                clf = KMeans(n_clusters = n)
            elif sys.argv[2] == 'gmm':
                clf = GMM(n_components = n)
            clf.fit(train_y)
            joblib.dump(clf, sys.argv[1]+'_'+sys.argv[2] +'_'+str(n)+'.pkl')
            print sys.argv[2] + str(n)
