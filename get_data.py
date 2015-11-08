import Quandl
import csv
import pandas

with open("WIKI-datasets-codes.csv", "rb") as data_list_file:
    data_list = csv.reader(data_list_file, delimiter=",")
    for row in data_list:
        print row
        mydata = Quandl.get(row[0],authtoken="KMZC_mVMw8iERbppDmQC")
        file_name = row[0].replace('/','-')
        mydata.to_csv(file_name+'.csv')
