import sys
from pymongo import MongoClient
import numpy as np
import copy

def Get_data_local(coll):
    '''Get data from collection local'''
    variable = []
    for i in range(1, coll.count()+1):
        doc = coll.find_one({'data_id':i})
        variable.append(doc)
    return variable

def Remove_list(list1, list2):
    '''remove element in list 2 from list1'''
    for item in list2:
        if item in list1:
            list1.remove(item)
    return list1


def Get_correl(variable, y_name):
    '''Compute correlation of the variables with y_name'''
    name_list_unwanted = ['_id', 'data_id', 'Date']
    name_list = variable[0].keys()
    name_list = Remove_list(name_list, name_list_unwanted)
    corr = {}
    for name in name_list:
        corr[name] = 0
    if y_name in name_list:
        for item in variable:
            for name in name_list:
                corr[name] += item[name] * item[y_name]
    else:
        print y_name+' is not a variable'
    return corr


if __name__ == '__main__':
    mongoclient = MongoClient()
    db = mongoclient.process_data
    coll_name = sys.argv[1] + '_norm'
    if coll_name in db.collection_names():
        coll = db[coll_name]
        variable = Get_data_local(coll)
        corr = Get_correl(variable, sys.argv[2])
        file_name = coll_name + '_corr_' + sys.argv[2] + '.txt'
        f = open(file_name, 'wb')
        for key in sorted(corr.keys()):
            print>>f, key, corr[key]
    else:
        print coll_name + ' is not in the db'
