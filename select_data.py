##--
# find the stock with maximum data
##--
import sys
import operator
import pickle
from pymongo import MongoClient

def Get_Find_Max(db):
    ''' sorted the stock with its data number '''
    collection_list = db.collection_names()
    stock_dict = {}
    ##--collect stock data
    for coll_name in collection_list[1:]:
        stock_dict[coll_name] = db[coll_name].count()
    return sorted(stock_dict.items(), key=operator.itemgetter(1), reverse=True)


    
if __name__ == '__main__':
    ##--connect to mongo client and database
    mongo_client = MongoClient()
    db = mongo_client.raw_data
    ##--get stock number dictionary
    stock_dict = Get_Find_Max(db)
    ##--output
    with open(sys.argv[1], 'wb') as f:
        if len(sys.argv) == 2 or sys.argv[2] == 'all':
            for i in range(0,len(stock_dict)):
                print>>f, stock_dict[i]
        else:
            for i in range(0, min(n, len(stock_dict))):
                print>>f, stock_dict[i]
