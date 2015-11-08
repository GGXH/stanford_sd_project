##--
# put data into mongo db
# each stock is a table
##--
import glob, os
import csv
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient

def Clean_col(col,value_list):
    '''Clean data set if value of value list is None'''
    for doc in col.find():
        for key in value_list:
            if not doc[key]:
                col.remove({'_id':doc['_id']})
                print 'Remove '
                print doc
                break

mongo_client = MongoClient()
db = mongo_client.raw_data

##os.chdir("..")

for file in glob.glob("*.csv"):
    table_name = file.split(".")[0].split("-")[1]
    print table_name
    db[table_name].drop()
    csvfile = open(file,'r')
    data = csv.DictReader(csvfile)
    ID = 0
    for each_row in data:
        adj_row = {}
        ID += 1
        for key, value in each_row.iteritems():
            adj_key = key.replace(".","")
            if key != 'Date':
                try:
                    adj_row[adj_key] = float(value)
                except:
                    adj_row[adj_key] = value
            else:
                adj_row[adj_key] = value
        adj_row['data_id'] = ID
        if None not in each_row.values() and '' not in each_row.values():
            db[table_name].insert(adj_row)
        else:
            print each_row
            print 'is skipped'
            ID -= 1




