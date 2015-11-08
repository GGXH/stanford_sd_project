##--
# put data into mongo db
# each stock is a table
##--
import glob, os
import csv
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient

mongo_client = MongoClient()
db = mongo_client.raw_data

print db.CUTR.find_one()
