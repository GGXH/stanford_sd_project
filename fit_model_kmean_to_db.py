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


def remark_y(y_km, no_limit=1000):
    '''remakr all the point if number in a cluster is less than no_limit, mark it as the nearest group'''
    ##--collect all the number of point in each cluster
    y_no = {}
    for i in y_km:
        y_no[i] = y_no.get(i,0) + 1
    ##--if there are more than 1000 point in one cluster, keep as it is.  Otherwise group it to the nearest cluster
    y_marker = {}
    for i in y_no.keys():
        if y_no[i] > no_limit:
            y_marker[i] = i
        else:
            min_dist = 100
            min_i = -1
            for j in range(0,km.cluster_centers_.shape[0]):
                if i != j and y_no[j] > no_limit and min_dist > abs(km.cluster_centers_[i][0]-km.cluster_centers_[j][0]):
                    min_i = j
                    min_dist = abs(km.cluster_centers_[i][0]-km.cluster_centers_[j][0])
            y_marker[i] = min_i
    print y_no
    print y_marker
    ##--remark all the group
    for i in range(0,y_km.shape[0]):
        y_km[i] = y_marker[y_km[i]]    
    ##--return
    return y_km


def Put_data_into_db(coll, x, var_name):
    '''put the data into db with name var_name'''
    for i in range(1, x.shape[0]+1):
        doc = coll.find_one({'data_id': i})
        coll.update({'_id':doc['_id']}, {'$set':{var_name:int(x[i-1])}})
        #print type(int(x[i-1]))
        #sys.exit(0)


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
        ##---
        for n in [2, 3, 4, 5]:
            km = joblib.load(sys.argv[1]+'_kmeans_' + str(n)  +'.pkl')
            y_km = km.predict(train_y)
            y_km = remark_y(y_km)
            var_name = 'price_diff_km_' + str(n)
            print set(y_km)
            print train_y[0:10]
            print km.cluster_centers_
            print y_km[0:10]
            Put_data_into_db(coll, y_km, var_name)
            #doc = coll.find_one()
            #print doc
