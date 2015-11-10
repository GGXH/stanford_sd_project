##--
# normalization data and put data into a new table
##--
import sys
from pymongo import MongoClient

def Get_Col_Name_list(coll, data_to_remove=None):
    ''' Get Column Name list of a collection '''
    record = coll.find_one()
    name_list = record.keys()
    name_list.pop(name_list.index('_id'))
    if data_to_remove:
        for data in data_to_remove:
            name_list.remove(data)
    return name_list


def Collect_data_collection(coll):
    ''' collect all the data from a collection  '''
    data_list = []
    for doc in coll.find():
        del doc['_id']
        data_list.append(doc)
    return data_list

def Get_data_dict(coll, key):
    '''get data of a key from a dictionary'''
    value_list = []
    for doc in coll:
        value_list.append(doc[key])
    return value_list

def Get_Mean(list):
    '''find mean value of a list'''
    return sum(list) / len(list)

def Stand_derivation(list, mean=None):
    '''find the standard derivation'''
    if not mean:
        mean = Get_Mean(list)
    return ( sum( [ ( x - mean )**2 for x in list ] ) / len(list) )**0.5


def Normalize_data(data_list, name_list):
    '''Normalize data by less the mean and divided by standard deviation'''
    for name in name_list:
        value_list = Get_data_dict(data_list, name)
        mean = Get_Mean(value_list)
        sd = Stand_derivation(value_list, mean)
        print name, mean, sd
        for i in range(0, len(data_list)):
            data_list[i][name] -= mean
            #if sd != 0:
            data_list[i][name] /= sd
    print 'Normalized data'



def Put_data_db(db, name, data_list):
    '''put a data list into a db with collection as name'''
    for data in data_list:
        db[name].insert(data)
    print 'Insert all the data in db as collection '+name

    
if __name__ == '__main__':
    ##--connect to mongo client and database
    mongo_client = MongoClient()
    db = mongo_client.raw_data
    db_process = mongo_client.process_data
    data_not_normal = ['data_id', 'Date']
    ##--
    if sys.argv[1] in db.collection_names():
        coll = db[sys.argv[1]]
        print coll.count()
        coll_name_list = Get_Col_Name_list(coll, data_not_normal)
        data_list = Collect_data_collection(coll)
        Normalize_data(data_list, coll_name_list)
        new_name = sys.argv[1]+'_norm'
        db[new_name].drop()
        Put_data_db(db_process, new_name, data_list)
        ##--
        doc = db_process[new_name].find_one()
        print doc
        print db_process[new_name].count()
