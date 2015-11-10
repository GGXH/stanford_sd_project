import sys
from pymongo import MongoClient

def Get_stock_movement(coll, movement_name):
    '''Get movement 1, if close > pre close, movement = 1; if close <= pre close, movement = -1'''
    for i in range(1, coll.count()+1):
        doc = coll.find_one({'data_id':i})
        if i != coll.count():
            doc_future = coll.find_one({'data_id':i+1})
            if doc['Close'] < doc_future['Close']:
                movement = 1
            else:
                movement = -1
        else:
            movment = -1
        coll.update({'_id':doc['_id']},{'$set':{movement_name: movement}})


def Get_close_price_diff(coll):
    '''Get price difference from previous close'''
    for i in range(1, coll.count()+1):
        doc = coll.find_one({'data_id':i})
        if i != coll.count():
            doc_future = coll.find_one({'data_id':i+1})
            price_diff = doc_future['Close'] - doc['Close']
        else:
            price_diff = 0
        coll.update({'_id':doc['_id']},{'$set':{'price_diff': price_diff}})


if __name__ == '__main__':
    mongoclient = MongoClient()
    db = mongoclient.process_data
    for coll_name in db.collection_names()[1:]:
        print coll_name
        coll = db[coll_name]
        Get_stock_movement(coll, 'movement1')
        Get_close_price_diff(coll)
        print coll.find_one()['price_diff']
        print coll.find_one()['movement1']
        
        
