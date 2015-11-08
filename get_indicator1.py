##--
# to calculate stock indicator
##--
import sys
import copy
from pymongo import MongoClient

def Get_Cal_No(collection, sys_arg):
    ''' Calculate the number to update '''
    doc_num = db[sys_argv[0]].count()
    if len(sys_argv) < 3 or sys_argv[1] == 'all':
        n_to_cal = db[sys_argv[0]].count()
    else:
        if sys_argv[1] <= doc_num:
            n_to_cal = sys_argv[2]
        else:
            print 'Document number requested is greater than document number in collection and will use document number in collection!'
            n_to_cal = doc_num
    return n_to_cal

def Clean_col(col,value_list):
    '''Clean data set if value of value list is None'''
    for doc in col.find():
        for key in value_list:
            if not doc[key]:
                col.remove({'_id':doc['_id']})
                print 'Remove '
                print doc
                break

def Find_Max_Index(list):
    ''' find the index of the highest value'''
    return list.index(max(list))

def Find_Min_Index(list):
    '''find the index of the lowest value'''
    return list.index(min(list))

def Get_Mean(list):
    '''find mean value of a list'''
    return sum(list) / len(list)

def Stand_derivation(list):
    '''find the standard derivation'''
    mean = Get_Mean(list)
    return ( sum( [ ( x - mean )**2 for x in list ] ) / len(list) )**0.5


def Get_ADL(col):
    ''' Calculate Accumulation Distribution Line (ADL)
    1. Money Flow Multiplier = [(Close - Low) - (High - Close)] / ( high - low)
    2. Money Flow Volume = Money Flow Multiplier * volume for the period
    3. ADL = Previous ADL + Current Period's Money Flow Volume'''
    ADL = 0
#    Clean_col(col, ['High', 'Low', 'Close'])
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if doc:
            MFM = 0
            print doc
            if  float(doc['High']) != float(doc['Low']):
                MFM = ( ( float(doc['Close']) - float(doc['Low']) ) - ( float(doc['High']) - float(doc['Close']) ) ) / ( float(doc['High']) - float(doc['Low']) ) 
                MFV = MFM * float(doc['Volume'])
                ADL = ADL + MFV
                col.update({'_id':doc['_id']},{'$set':{'ADL':ADL}})
    print 'Got accumulation distribution'



def Get_Aroon(col):
    '''Calculate Aroon
    Aroon-Up = ( ( 25 - Days Since 25-day High ) / 25 ) * 100
    Aroon-Down = ( ( 25 - Days Since 25-day Low ) / 25 ) * 100'''
    high_list = []
    low_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        high_list.insert(0, doc['High'])
        low_list.insert(0,doc['Low'])
        Aroon_up = ( ( len(high_list) - Find_Max_Index(high_list) * 1.0 ) / len(high_list) ) * 100.
        Aroon_down = ( ( len(low_list) - Find_Min_Index(low_list) * 1. ) / len(low_list) ) * 100.
        Aroon_osc = Aroon_up - Aroon_down
        col.update({'_id':doc['_id']},{'$set':{'Aroon_up':Aroon_up, 'Aroon_down':Aroon_down, 'Aroon_osc': Aroon_osc}})
        if len(high_list) == 26:
            high_list.pop()
            low_list.pop()
    print 'Got Aroon'


def Get_TR(col):
    '''Get true range of stock price'''
    old_doc = {}
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if len(old_doc) == 0:
            TR = doc['High'] - doc['Low']
        else:
            TR = max(doc['High']-old_doc['Close'], old_doc['Close']-doc['Low'], doc['High']-doc['Low'])
        old_doc = copy.deepcopy(doc)
        col.update({'_id':doc['_id']},{'$set':{'TR':TR}})
    print 'Got true range of stock price'


def Get_DM(col):
    ''' Get directional movement'''
    old_doc = {}
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        DMP = 0
        DMN = 0
        if len(old_doc) != 0:
            if doc['High'] > old_doc['High']:
                DMP = doc['High'] - old_doc['High']
            if old_doc['Low'] > doc['Low']:
                DMN = old_doc['Low'] - doc['Low']
        old_doc = copy.deepcopy(doc)
        col.update({'_id':doc['_id']},{'$set':{'DMP':DMP, 'DMN':DMN}})
    print 'Got directional movement'


def Get_14D_WilderSM1(col, value):
    '''Get 14-day wilder's smoothing of a value: method 1'''
    value_list = []
    new_name = value+'_14WSM'
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if len(value_list) < 15:
            value_list.insert(0,doc[value])
            value_14 = sum(value_list) / len(value_list)
        else:
            value_14 = old_value_14 * 13. / 14. + doc[value]
        old_value_14 = value_14
        col.update({'_id':doc['_id']},{'$set':{new_name: value_14}})
    print 'Got '+new_name

def Get_14D_WilderSM2(col, base, name):
    '''Get 14-day wilder's smoothing of a value: method 1'''
    value_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if len(value_list) < 15:
            value_list.insert(0,doc[base])
            value_14 = sum(value_list) / len(value_list)
        else:
            value_14 = ( old_value_14 * 13. + doc[base] ) / 14.
        old_value_14 = value_14
        col.update({'_id':doc['_id']},{'$set':{name: value_14}})    


def Get_DI14_DX(col):
    '''Get Plus/Minus directional indicator and directional movement index'''
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        DI14P = doc['DMP_14WSM'] / doc['TR_14WSM'] * 100
        DI14N = doc['DMN_14WSM'] / doc['TR_14WSM'] * 100
        DX = 0
        if DI14P != 0 or DI14N != 0:
            DX = abs(( DI14P - DI14N ) / ( DI14P + DI14N ))
        col.update({'_id':doc['_id']},{'$set':{'DI14P': DI14P, 'DI14N': DI14N, 'DX': DX}})
    print 'Got Plus/Minus directional indicator and directional movement index'


def Get_ADX(col):
    '''Get average directional index'''
    ##--get true range
    Get_TR(col)
    ##--get directional movement
    Get_DM(col)
    ##--get 14-day smoothed TR, DMP, and DMN
    Get_14D_WilderSM1(col,'TR')
    Get_14D_WilderSM1(col,'DMP')
    Get_14D_WilderSM1(col,'DMN')
    ##--get directional indicator and directional movement index
    Get_DI14_DX(col)
    ##--
    Get_14D_WilderSM2(col, 'DX', 'ADX')
    print 'Got average directional index'


def Get_ATR(col):
    '''Get average true range'''
    Get_14D_WilderSM2(col, 'TR', 'ATR')
    print 'Got average true range'


def Get_BB(col):
    '''Get bollinger band'''
    value_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value_list.insert(0,doc['Close'])
        MBB = Get_Mean(value_list)
        stand_der = Stand_derivation(value_list)
        UBB = MBB + 2 * stand_der
        LBB = MBB - 2 * stand_der
        BBB = ( UBB - LBB ) / MBB * 100.
        col.update({'_id':doc['_id']},{'$set':{'MBB': MBB, 'UBB': UBB, 'LBB': LBB, 'BBB': BBB}})
        if len(value_list) == 20:
            value_list.pop()
    print 'Got bolling bandwidth'

def Get_Bindicator(col):
    '''Get %B indicator'''
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        PBI = 0
        if doc['UBB'] != doc['LBB']:
            PBI = ( doc['Close'] - doc['LBB'] ) / ( doc['UBB'] - doc['LBB'] )
        col.update({'_id':doc['_id']},{'$set':{'PBI': PBI}})
    print 'Got %B indicator'

def Get_CGI(col):
    '''Get commodity channel index'''
    value_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'date_id':i})
        print doc
        print len(value_list)
        TP = ( doc['High'] + doc['Low'] + doc['Close'] ) / 3.
####---unfinished---#########

        value_list.insert(0,doc['Close'])
        MBB = Get_Mean(list)
        stand_der = Stand_derivation(list)
        UBB = MBB + 2 * stand_der
        LBB = MBB - 2 * stand_der
        BBB = ( UBB - LBB ) / MBB * 100.
        col.update({'_id':doc['_id']},{'$set':{'MBB': MBB, 'UBB': UBB, 'LBB': LBB, 'BBB': BBB}})
        if len(value_list) == 20:
            value_list.pop()
    print 'Got bolling bandwidth'    



if __name__ == '__main__':
##--connect to mongo client and database
    mongo_client = MongoClient()
    db = mongo_client.raw_data
    collection_list = db.collection_names()
    if sys.argv[1] == 'all':
        ##--update all collections
        for coll_name in collection_list[1:]:
            print coll_name
            coll = db[coll_name]
            Get_ADL(coll)
            Get_Aroon(coll)
            #Get_TR(coll)
            #Get_DM(coll)
            Get_ADX(coll)
            Get_ATR(coll)
            Get_BB(coll)
            Get_Bindicator(coll)            
    else:
        ##--update one collection
        if sys.argv[1] in collection_list:
            Get_ADL(db[sys.argv[1]])
            Get_Aroon(db[sys.argv[1]])
            #Get_TR(db[sys.argv[1]])
            #Get_DM(db[sys.argv[1]])
            Get_ADX(db[sys.argv[1]])
            Get_ATR(db[sys.argv[1]])
            Get_BB(db[sys.argv[1]])
            Get_Bindicator(db[sys.argv[1]])
        else:
            print 'Collection'+sys.argv[1]+' does not exist!'

