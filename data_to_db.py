import csv
from pymongo import MongoClient
import pymongo

client = MongoClient()

db = client['test-db']

#db['test-col'].drop()

#collection = db['test-col']

#csvfile = open('WIKI-AAL.csv','r')

#fieldnames = ("Date", "Open", "High", "Low", "Close", "Volume", "Ex-Dividend", "Split Ratio", "Adj Open", "Adj High", "Adj Low", "Adj Close", "Adj Volumn")

#reader = csv.DictReader(csvfile, fieldnames)

#for row in reader:
#    print row
#    collection.insert_one(row)


#print db.getCollectionNames()

print db.collection_names(include_system_collections=True)

print db['test-col'].find_one()
