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

def Get_WMA(list):
    '''Get weighted aveerage'''
    mean_weight = sum([i for i in range(1,len(list)+1)])
    return sum( [ list[i]*(i+1) for i in range(0,len(list)) ] ) / mean_weight

def Get_EMA(close, old, n, list=[]):
    '''Get exponential moving average over n period'''
    if len(list) < n+1 and len(list) != 0:
        return Get_Mean(list)
    else:
        multiplier = 2. / ( n + 1. )
        return ( close - old ) * multiplier + old

def Get_CSF(close, old, n, list=[]):
    '''Get exponential moving average over n period'''
    if len(list) < n+1 and len(list) != 0:
        return Get_Mean(list)
    else:
        multiplier = 2. / n
        return ( close - old ) * multiplier + old

def Get_sign_mean(list, sgn):
    '''Get signed average of list, return the mean value of element with the same sign as sng'''
    sum = 0
    for x in list:
        if x*sgn >= 0:
            sum += x
    return sum


def Get_ADL(col):
    ''' Calculate Accumulation Distribution Line (ADL)
    1. Money Flow Multiplier = [(Close - Low) - (High - Close)] / ( high - low)
    2. Money Flow Volume = Money Flow Multiplier * volume for the period
    3. ADL = Previous ADL + Current Period's Money Flow Volume'''
    ADL = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        MFM = 0
        if  float(doc['High']) != float(doc['Low']):
            MFM = ( ( float(doc['Close']) - float(doc['Low']) ) - ( float(doc['High']) - float(doc['Close']) ) ) / ( float(doc['High']) - float(doc['Low']) ) 
        MFV = MFM * float(doc['Volume'])
        ADL = ADL + MFV
        col.update({'_id':doc['_id']},{'$set':{'ADL':ADL, 'MFM': MFM, 'MFV': MFV}})
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
    old_close = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if old_close == 0:
            TR = doc['High'] - doc['Low']
        else:
            TR = max(doc['High'], old_close) - min(old_close, doc['Low'])
        old_close = doc['Close']
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
        doc = col.find_one({'data_id':i})
        TP = ( doc['High'] + doc['Low'] + doc['Close'] ) / 3.
        value_list.insert(0,TP)
        sd = Stand_derivation(value_list)
        if sd != 0:
            CGI = ( TP - Get_Mean(value_list) ) / 0.015 / sd
        else:
            CGI = 0
        if len(value_list) == 20:
            value_list.pop()
        col.update({'_id':doc['_id']},{'$set':{'CGI': CGI, 'TP': TP}})
    print 'Got commodity channel index'    


def Get_ROC_n(col, n):
    '''Get rate of change with n period'''
    value_list = []
    name = 'ROC_' + str(n)
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value_list.insert(0,doc['Close'])
        ROC = ( doc['Close'] - value_list[-1] ) / value_list[-1] * 100
        if len(value_list) == n:
            value_list.pop()
        col.update({'_id':doc['_id']},{'$set':{name: ROC}})
    print 'Got rate of change with '+ str(n) +' period'


def Get_CopCv(col):
    '''Get coppock curve'''
    Get_ROC_n(col,11)
    Get_ROC_n(col,14)
    value_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value_list.append(doc['ROC_14'])
        CopCv = Get_WMA(value_list) + doc['ROC_11']
        if len(value_list) == 10:
            value_list.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'CopCv': CopCv}})
    print 'Got coppock curve'    


def Get_CMF(col):
    '''Get Chaikin Money Flow'''
    MFV_list = []
    vol_list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        MFV_list.append(doc['MFV'])
        vol_list.append(doc['Volume'])
        Mean_volume = Get_Mean(vol_list)
        CMF_20 = 0
        if Mean_volume != 0:
            CMF_20 = Get_Mean(MFV_list) / Mean_volume
        if len(MFV_list) == 20:
            MFV_list.pop(0)
            vol_list.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'CMF_20': CMF_20}})
    print 'Got Chaikin Money Flow' 


def Get_Chaikin_osc(col):
    '''Get Chaikin oscillator'''
    ADL_list = []
    ADL_10_old = 0
    ADL_3_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i <= 10:
            ADL_list.append(doc['ADL'])
            ADL_10 = Get_EMA(doc['Close'], ADL_10_old, 10, ADL_list)
        else:
            ADL_10 = Get_EMA(doc['Close'], ADL_10_old, 10)
        if i <= 3:
            ADL_3 = Get_EMA(doc['Close'], ADL_3_old, 3, ADL_list)
        else:
            ADL_3 = Get_EMA(doc['Close'], ADL_3_old, 3)           
        ADL_10_old = ADL_10
        ADL_3_old = ADL_3
        Chai_osc = ADL_3 - ADL_10
        col.update({'_id':doc['_id']},{'$set':{'Chain_osc': Chai_osc}})
    print 'Get Chaikin osc'


def Get_PMO(col):
    '''Get Price Momemtum Oscillator'''
    list = []
    n = 35
    CS_35_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            value = 0.
        else:
            value = ( doc['Close'] / old_value * 100. ) - 100.
        old_value = doc['Close']
        if i < n:
            list.append(value)
            CS_35 = Get_CSF(value, CS_35_old, n, list)
        else:
            CS_35 = Get_CSF(value, CS_35_old, n)
        CS_35_old = CS_35
        col.update({'_id':doc['_id']},{'$set':{'PMO_line': CS_35 * 10.}})
    list = []
    n = 20
    CS_20_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i < n:
            list.append(doc['PMO_line'])
            CS_20 = Get_CSF(doc['PMO_line'], CS_20_old, n, list)
        else:
            CS_20 = Get_CSF(doc['PMO_line'], CS_20_old, n)
        CS_20_old = CS_20
        col.update({'_id':doc['_id']},{'$set':{'PMO_line': CS_20}})
    print 'Got Price Momentum Oscillator line'
    list = []
    n = 10
    PMO_10_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i < n:
            list.append(doc['PMO_line'])
            PMO_10 = Get_EMA(doc['PMO_line'], PMO_10_old, n, list)
        else:
            PMO_10 = Get_EMA(doc['PMO_line'], PMO_10_old, n)
        PMO_10_old = PMO_10
        col.update({'_id':doc['_id']},{'$set':{'PMO_sign_line': PMO_10}})
    print 'Got price momentum oscillator signal line'


def Get_DPO(col, n):
    '''Get Detrended price oscillator over n period'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        list.insert(0, doc['Close'])
        if i == 1:
            DPO = 0
        else:
            j = int(len(list)/2+1)
            if j >= len(list):
                j = len(list) - 1
            DPO = list[j] - Get_Mean(list)
        col.update({'_id':doc['_id']},{'$set':{'DPO': DPO}})
        if len(list) == n:
            list.pop()
    print 'Got Detrended Price Oscillator over '+str(n)+' period'


def Get_EMV(col):
    '''Get Ease of Movement'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            distance_moved = 0
        else:
            distance_moved = ( ( doc['High'] + doc['Low'] )  - ( old_high - old_low ) ) 
        old_high = doc['High']
        old_low = doc['Low']
        if doc['High'] == doc['Low']:
            box_ratio = 0
        else:
            box_ratio = doc['Volume'] / ( doc['High'] - doc['Low'] )
        if box_ratio != 0:
            EMV1 = distance_moved / box_ratio
        else:
            EMV1 = 0
        list.insert(0,EMV1)
        EMV14 = Get_Mean(list)
        col.update({'_id':doc['_id']},{'$set':{'EMV': EMV14}})
        if len(list) == 14:
            list.pop()
    print 'Got Ease of Movement'
        

def Get_FI(col):
    '''Get Force index'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            FI1 = 0
        else:
            FI1 = ( doc['Close'] - old_close ) * doc['Volume']
        old_close = doc['Close']
        list.insert(0,FI1)
        FI13 = Get_Mean(list)
        col.update({'_id':doc['_id']},{'$set':{'FI1': FI1, 'FI13': FI13}})
        if len(list) == 13:
            list.pop()
    print 'Got Force index'    


def Get_MI(col):
    '''Get Mass index'''
    single_EMA = []
    double_EMA = []
    ##--Get single EMA
    list = []
    n = 9
    single_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['High'] - doc['Low']
        if i < n:
            list.append(value)
            single = Get_EMA(value, single_old, n, list)
        else:
            single = Get_EMA(value, single_old, n)
        single_old = single
        single_EMA.append(single)
    ##--Get double EMA
    list = []
    n = 9
    double_old = 0
    for i in range(1,col.count()+1):
        value = single_EMA[i-1]
        if i < n:
            list.append(value)
            double = Get_EMA(value, double_old, n, list)
        else:
            double = Get_EMA(value, double_old, n)
        double_old = double
        double_EMA.append(double)
    ##--Get EMA Ratio
    for i in range(1,col.count()+1):
        if double_EMA[i-1] != 0:
            single_EMA[i-1] = single_EMA[i-1] / double_EMA[i-1]
        else:
            single_EMA[i-1] = 0
    ##--Get Mass Index
    list = []
    n = 25
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = single_EMA[i-1]
        list.append(value)
        MI = sum(list)
        col.update({'_id':doc['_id']},{'$set':{'MI': MI}})
        if len(list) == 25:
            list.pop(0)
    print 'Got Mass index'


def Get_MACD_PPO(col):
    '''Get moving average convergence/divergence oscillator and Percentage price oscillator'''
    list12 = []
    list26 = []
    n12 = 12
    n26 = 26
    value12_old = 0
    value26_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['Close']
        ##--get 12-day EMA
        if i < n12:
            list12.append(value)
            value12 = Get_EMA(value, value12_old, n12, list12)
        else:
            value12 = Get_EMA(value, value12_old, n12)
        value12_old = value12
        ##--get 26-day EMA
        if i < n26:
            list26.append(value)
            value26 = Get_EMA(value, value26_old, n26, list26)
        else:
            value26 = Get_EMA(value, value26_old, n26)
        value26_old = value26
        ##--MACD line
        MACD_line = value12 - value26
        PPO = MACD_line / value26 * 100
        col.update({'_id':doc['_id']},{'$set':{'MACD_line': MACD_line, 'PPO': PPO}})
    print 'Got MACD and PPO line'
    ##--Get MACD histogram
    list12 = []
    list26 = []
    n12 = 9
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['MACD_line']
        value1 = doc['PPO']
        ##--get 9-day EMA
        if i < n12:
            list12.append(value)
            list26.append(value1)
            value12 = Get_EMA(value, value12_old, n12, list12)
            value26 = Get_EMA(value1, value26_old, n12, list26)
        else:
            value12 = Get_EMA(value, value12_old, n12)
            value26 = Get_EMA(value1, value26_old, n12)
        value12_old = value12
        value26_old = value26
        MACD_hist = value - value12
        PPO_hist = value1 - value26
        col.update({'_id':doc['_id']},{'$set':{'MACD_hist': MACD_hist, 'PPO_hist': PPO_hist}})
    print 'Got MACD and PPO histogram'


def Get_MFI(col):
    '''Get money flow index'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        list.append(doc['TP']*doc['Volume'])
        PMF14 = Get_sign_mean(list, 1)
        NMF14 = Get_sign_mean(list, -1) * -1
        if NMF14 != 0:
            MFR = PMF14 / NMF14
            MFI = 100 - 100 / ( 1. + MFR )
        else:
            MFI = 100
        col.update({'_id':doc['_id']},{'$set':{'MFI': MFI}})
        if len(list) == 14:
            list.pop(0)
    print 'Got Money flow index'    


def Get_NVI(col):
    '''Get negative volume index'''
    list = []
    n = 255
    value_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            CNVI = 1000
        else:
            if doc['Volume'] < old_vol:
                SPX = ( doc['Close'] - old_close ) / old_close * 100.
            else:
                SPX = 0
            CNVI += SPX
        old_vol = doc['Volume']
        old_close = doc['Close']
        if i < n:
            list.append(CNVI)
            value = Get_EMA(CNVI, value_old, n, list)
        else:
            value = Get_EMA(CNVI, value_old, n)
        old_value = value
        col.update({'_id':doc['_id']},{'$set':{'NVI': value}})
    print 'Got negative volume index'


def Get_OBV(col):
    '''Get on balance volume'''
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            OBV = doc['Volume']
        else:
            if doc['Close'] > old_close:
                OBV += doc['Volume']
            elif doc['Close'] < old_close:
                OBV -= doc['Volume']
        old_close = doc['Close']
        old_vol = doc['Volume']
        col.update({'_id':doc['_id']},{'$set':{'OBV': OBV}})
    print 'Got on balance volume'   


def Get_PVO(col):
    '''Get Percentage volume oscillator'''
    list12 = []
    list26 = []
    n12 = 12
    n26 = 26
    value12_old = 0
    value26_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['Volume']
        ##--get 12-day EMA
        if i < n12:
            list12.append(value)
            value12 = Get_EMA(value, value12_old, n12, list12)
        else:
            value12 = Get_EMA(value, value12_old, n12)
        value12_old = value12
        ##--get 26-day EMA
        if i < n26:
            list26.append(value)
            value26 = Get_EMA(value, value26_old, n26, list26)
        else:
            value26 = Get_EMA(value, value26_old, n26)
        value26_old = value26
        ##--MACD line
        PVO_line = value12 - value26
        col.update({'_id':doc['_id']},{'$set':{'PVO_line': PVO_line}})
    print 'Got PVO line'
    ##--Get MACD histogram
    list12 = []
    n12 = 9
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['PVO_line']
        ##--get 9-day EMA
        if i < n12:
            list12.append(value)
            value12 = Get_EMA(value, value12_old, n12, list12)
        else:
            value12 = Get_EMA(value, value12_old, n12)
        value12_old = value12
        PVO_hist = value - value12
        col.update({'_id':doc['_id']},{'$set':{'PVO_hist': PVO_hist}})
    print 'Got PVO histogram'


def Get_n_SMA(col, n, base, target):
    '''Get n period SMA of base and update to target'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        list.insert(0, doc[base])
        target_value = Get_Mean(list)
        col.update({'_id':doc['_id']},{'$set':{target: target_value}})
        if len(list) == n:
            list.pop()
    print 'Got '+target


def Get_KST(col):
    '''Get Know sure thing'''
    Get_ROC_n(col, 10)
    Get_ROC_n(col, 15)
    Get_ROC_n(col, 20)
    Get_ROC_n(col, 30)
    ##--
    Get_n_SMA(col, 10, 'ROC_10', 'RCMA1')
    Get_n_SMA(col, 10, 'ROC_15', 'RCMA2')
    Get_n_SMA(col, 10, 'ROC_20', 'RCMA3')
    Get_n_SMA(col, 15, 'ROC_30', 'RCMA4')
    ##--
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['RCMA1'] + doc['RCMA2'] * 2 + doc['RCMA3'] * 3 + doc['RCMA4'] * 4
        col.update({'_id':doc['_id']},{'$set':{'KST': value}})
    ##--
    Get_n_SMA(col, 9, 'KST', 'KST_sng_line')
    

def Get_SpecK(col):
    '''Get special K'''
    Get_ROC_n(col, 10)
    Get_ROC_n(col, 15)
    Get_ROC_n(col, 20)
    Get_ROC_n(col, 30)    
    Get_ROC_n(col, 40)
    Get_ROC_n(col, 65)
    Get_ROC_n(col, 75)
    Get_ROC_n(col, 100)
    Get_ROC_n(col, 195)
    Get_ROC_n(col, 265)
    Get_ROC_n(col, 390)
    Get_ROC_n(col, 530)
    ##--
    Get_n_SMA(col, 10, 'ROC_10', 'SK1')
    Get_n_SMA(col, 10, 'ROC_15', 'SK2')
    Get_n_SMA(col, 10, 'ROC_20', 'SK3')
    Get_n_SMA(col, 15, 'ROC_30', 'SK4')
    Get_n_SMA(col, 50, 'ROC_40', 'SK5')
    Get_n_SMA(col, 65, 'ROC_65', 'SK6')
    Get_n_SMA(col, 75, 'ROC_75', 'SK7')
    Get_n_SMA(col, 100, 'ROC_100', 'SK8')
    Get_n_SMA(col, 130, 'ROC_195', 'SK9')
    Get_n_SMA(col, 130, 'ROC_265', 'SK10')
    Get_n_SMA(col, 130, 'ROC_390', 'SK11')
    Get_n_SMA(col, 195, 'ROC_530', 'SK12')
    ##--
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['SK1'] + doc['SK2'] * 2 + doc['SK3'] * 3 + doc['SK4'] * 4 + doc['SK5'] + doc['SK6'] * 2 + doc['SK7'] * 3 + doc['SK8'] * 4 + doc['SK9'] + doc['SK10'] * 2 + doc['SK11'] * 3 + doc['SK12'] * 4
        col.update({'_id':doc['_id']},{'$set':{'SpecK': value}})
    print 'Got special K'


def Get_RSI(col):
    '''Get relative strenght index'''
    list = []
    n = 14
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            value = 0
        else:
            value = doc['Close'] - old_close
        old_close = doc['Close']
        if i <= 14:
            list.append(value)
            gain = Get_sign_mean(list, 1)
            loss = Get_sign_mean(list, -1) * -1
        else:
            if value > 0:
                gain = ( gain * 13 + value ) / 14
            else:
                loss = ( loss * 13 - value ) / 14
        if loss == 0:
            RS = 0
            RSI = 100
        else:
            RS = gain / loss
            RSI = 100. - 100. / ( 1. + RS )
        col.update({'_id':doc['_id']},{'$set':{'RSI': RSI, 'RS': RS}})
    print 'Got relative strength and index'


def Get_SO(col):
    '''Get stochastic oscillator'''
    list = []
    list_k = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i <= 2:
            perc_K = 0
        else:
            perc_K = ( doc['Close'] - min(list) ) / ( max(list) - min(list) ) * 100
        list.append(doc['Close'])
        if len(list) == 15:
            list.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'perc_K': perc_K}})
        list_k.append(perc_K)
        perc_D = Get_Mean(list_k)
        col.update({'_id':doc['_id']},{'$set':{'perc_D': perc_D}})
        if len(list_k) == 3:
            list_k.pop(0)
    print 'Got stochastic oscillator'


def Get_StochRSI(col):
    '''Get stochastic RSI'''
    list = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1 or max(list) == min(list):
            value = 0
        else:
            value = ( doc['RSI'] - min(list) ) / ( max(list) - min(list) ) * 100
        list.append(doc['RSI'])
        if len(list) == 15:
            list.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'stochRSI': value}})
    print 'Got stochastic RSI'



def Get_TRIX(col):
    '''Get TRIX'''
    single_EMA = []
    double_EMA = []
    tri_EMA = []
    ##--Get single EMA
    list = []
    n = 15
    single_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        value = doc['Close']
        if i < n:
            list.append(value)
            single = Get_EMA(value, single_old, n, list)
        else:
            single = Get_EMA(value, single_old, n)
        single_old = single
        single_EMA.append(single)
    ##--Get double EMA
    list = []
    n = 15
    double_old = 0
    for i in range(1,col.count()+1):
        value = single_EMA[i-1]
        if i < n:
            list.append(value)
            double = Get_EMA(value, double_old, n, list)
        else:
            double = Get_EMA(value, double_old, n)
        double_old = double
        double_EMA.append(double)
    ##--Get triple EMA
    list = []
    n = 15
    tri_old = 0
    for i in range(1,col.count()+1):
        value = double_EMA[i-1]
        if i < n:
            list.append(value)
            tri = Get_EMA(value, tri_old, n, list)
        else:
            tri = Get_EMA(value, tri_old, n)
        tri_old = tri
        tri_EMA.append(tri)
    ##--Get Mass Index
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            value = 0
        else:
            value = ( tri_EMA[i-1] - tri_EMA[i-2] ) / tri_EMA[i-2]
        col.update({'_id':doc['_id']},{'$set':{'TRIX': value}})
    print 'Got TRIX'


def Get_TSI(col):
    '''Get TRIX'''
    single_EMA = []
    double_EMA = []
    single_EMA_A = []
    double_EMA_A = []
    ##--Get single EMA
    list = []
    n = 25
    single_old = 0
    single_A_old = 0
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            value = 0
        else:
            value = doc['Close'] - old_close
        old_close = doc['Close']
        if i < n:
            list.append(value)
            single = Get_EMA(value, single_old, n, list)
            single_A = Get_EMA(abs(value), single_A_old, n, list)
        else:
            single = Get_EMA(value, single_old, n)
            single_A = Get_EMA(abs(value), single_A_old, n)
        single_old = single
        single_A_old = single_A
        single_EMA.append(single)
        single_EMA_A.append(single_A)
    ##--Get double EMA
    list = []
    n = 13
    double_old = 0
    double_A_old = 0
    for i in range(1,col.count()+1):
        if i < n:
            list.append(value)
            double = Get_EMA(single_EMA[i-1], double_old, n, list)
            double_A = Get_EMA(single_EMA_A[i-1], double_A_old, n, list)
        else:
            double = Get_EMA(single_EMA[i-1], double_old, n)
            double_A = Get_EMA(single_EMA_A[i-1], double_A_old, n)
        double_old = double
        double_A_old = double_A
        double_EMA.append(double)
        double_EMA_A.append(double_A)
    ##--Get TSI
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        TSI = ( double_EMA[i-1] / double_EMA_A[i-1] ) * 100.
        col.update({'_id':doc['_id']},{'$set':{'TSI': TSI}})
    print 'Got TSI'


def Get_UlcerI(col):
    '''Get Ulcer index'''
    list_close = []
    list_pd = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})    
        list_close.append(doc['Close'])
        perc_d = ( doc['Close'] - max(list_close) ) / max(list_close) * 100.
        if len(list_close) == 14:
            list_close.pop(0)
        list_pd.append(perc_d**2)
        UlcerI = (Get_Mean(list_pd) )**0.5
        if len(list_pd) == 14:
            list_pd.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'UlcerI': UlcerI}})
    print 'Got Ulcer index'


def Get_UO(col):
    '''Get Ultimate Oscillator'''
    list_bp = []
    list_tr = []
    old_close = 1000000
    for i in range(1,col.count()+1):   
        doc = col.find_one({'data_id':i})
        BP = doc['Close'] - min(doc['Low'], old_close)
        col.update({'_id':doc['_id']},{'$set':{'BP': BP}})
        list_bp.insert(0,BP)
        list_tr.insert(0,doc['TR'])
        average7 = Get_Mean(list_bp[:min(len(list_bp),7)]) / Get_Mean(list_tr[:min(len(list_tr),7)])
        average14 = Get_Mean(list_bp[:min(len(list_bp),14)]) / Get_Mean(list_tr[:min(len(list_tr),14)])
        average28 = Get_Mean(list_bp[:min(len(list_bp),28)]) / Get_Mean(list_tr[:min(len(list_tr),28)])
        UO = 100 * ( 4 * average7 + 2 * average14 + average28 ) / 7.
        col.update({'_id':doc['_id']},{'$set':{'UO': UO, 'BP': BP}})
        if len(list_bp) == 28:
            list_bp.pop()
        if len(list_tr) == 28:
            list_tr.pop()
    print 'Got Ultimate Oscillator and buying pressure'


def Get_VI(col):
    '''Get vortex index'''
    list_pvm = []
    list_nvm = []
    list_tr = []
    for i in range(1,col.count()+1):   
        doc = col.find_one({'data_id':i})
        if i == 1:
            pvm = doc['High'] - doc['Low']
            nvm = doc['High'] - doc['Low']
        else:
            pvm = abs( doc['High'] - old_low )
            nvm = abs( doc['Low'] - old_high )
        old_low = doc['Low']
        old_high = doc['High']
        list_pvm.insert(0,pvm)
        list_nvm.insert(0,nvm)
        list_tr.insert(0,doc['TR'])
        pvi = Get_Mean(list_pvm) / Get_Mean(list_tr)
        nvi = Get_Mean(list_nvm) / Get_Mean(list_tr)
        col.update({'_id':doc['_id']},{'$set':{'PVI': pvi, 'NVI': nvi}})
        if len(list_pvm) == 14:
            list_pvm.pop()
            list_nvm.pop()
            list_tr.pop()
    print 'Got vortex indicator'    
    

def Get_WillR(col):
    '''Get Williams %R'''
    list_high = []
    list_low = []
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            perc_R = ( doc['High'] - doc['Close'] ) / ( doc['High'] - doc['Low'] ) * (-100)
        else:
            perc_R = ( max(list_high) - doc['Close'] ) / ( max(list_high) - min(list_low) ) * (-100)
        list_high.append(doc['High'])
        list_low.append(doc['Low'])
        if len(list_high) == 15:
            list_high.pop(0)
            list_low.pop(0)
        col.update({'_id':doc['_id']},{'$set':{'perc_R': perc_R}})
    print 'Got William %R'


def Get_x_n(col, base, n, m):
    '''Get previous n value averged over previous m value'''
    list = []
    list_avg = []
    name_list = []
    for i in range(1,n+1):
        name_list.append(base+'_'+str(m)+'_'+str(i))
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if i == 1:
            for j in range(1,n+1):
                list.append(doc[base])
            for j in range(1,m+1):
                list_avg.append(doc[base])
        else:
            list.insert(0, Get_Mean(list_avg))
            list_avg.insert(0,doc[base])
        for j in range(1,n+1):
            col.update({'_id':doc['_id']},{'$set':{name_list[j-1]: list[j-1]}})
        list.pop()
        if len(list_avg) > m:
            list_avg.pop()
    print 'Got '+' '.join(name_list)
            

def Get_FD_slope(col, base, n, order):
    '''Get slope with finite difference method'''
    name_list = []
    for i in range(1,order+1):
        name_list.append(base+'_'+str(n)+'_'+str(i))
    name = 'Slope_'+str(n)+'_'+str(order)
    for i in range(1,col.count()+1):
        doc = col.find_one({'data_id':i})
        if order == 1:
            slope = doc[base] - doc[name_list[0]]
        elif order == 2:
            slope = 2 * doc[base] - 3 * doc[name_list[0]] + doc[name_list[1]]
        elif order == 3:
            slope = ( 26 * doc[base] - 57 * doc[name_list[0]] + 42 * doc[name_list[1]] - 11 * doc[name_list[2]] ) / 6.
        elif order == 4:
            slope = ( 177. / 16 - 299. / 32 + 163. / 48 - 25. / 64 ) * doc[base] - 177. / 16 * doc[name_list[0]] + 299. / 32 * doc[name_list[1]] - 163. / 48 * doc[name_list[2]] + 25. / 64 * doc[name_list[3]]
        col.update({'_id':doc['_id']},{'$set':{name: slope}})
#        print doc
#        print doc[name]
#        if i > 5:
#            sys.exit(0)
    print 'Got ' + name


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
        list = sys.argv[1].split('+')
        print list
        #if sys.argv[1] in collection_list:
        for coll_name in list:
            print coll_name
            coll = db[coll_name]
            #Get_ADL(coll)
            #Get_Aroon(coll)
            #Get_TR(coll)
            #Get_DM(coll)
            #Get_ADX(coll)
            #Get_ATR(coll)
            #Get_BB(coll)
            #Get_Bindicator(coll)
            #Get_CGI(coll)
            #Get_CopCv(coll)
            #Get_CMF(coll)
            #Get_Chaikin_osc(coll)
            #Get_PMO(coll)
            #Get_DPO(coll, 20)
            #Get_EMV(coll)
            #Get_FI(coll)
            #Get_MI(coll)
            #Get_MACD_PPO(coll)
            #Get_MFI(coll)
            #Get_NVI(coll)
            #Get_OBV(coll)
            #Get_PVO(coll)
            #Get_KST(coll)
            #Get_SpecK(coll)
            #Get_RSI(coll)
            #Get_SO(coll)
            #Get_StochRSI(coll)
            #Get_TRIX(coll)
            #Get_TSI(coll)
            #Get_UlcerI(coll)
            #Get_UO(coll)
            #Get_VI(coll)
            #Get_WillR(coll)
            Get_x_n(coll, 'Close', 30, 1)
            Get_x_n(coll, 'Close', 30, 5)
            Get_x_n(coll, 'Close', 30, 10)
            Get_x_n(coll, 'Close', 30, 20)
            Get_FD_slope(coll, 'Close', 1, 1)
            Get_FD_slope(coll, 'Close', 1, 2)
            Get_FD_slope(coll, 'Close', 1, 3)
            Get_FD_slope(coll, 'Close', 1, 4)
            Get_FD_slope(coll, 'Close', 5, 1)
            Get_FD_slope(coll, 'Close', 5, 2)
            Get_FD_slope(coll, 'Close', 5, 3)
            Get_FD_slope(coll, 'Close', 5, 4)
            Get_FD_slope(coll, 'Close', 10, 1)
            Get_FD_slope(coll, 'Close', 10, 2)
            Get_FD_slope(coll, 'Close', 10, 3)
            Get_FD_slope(coll, 'Close', 10, 4)
            Get_FD_slope(coll, 'Close', 20, 1)
            Get_FD_slope(coll, 'Close', 20, 2)
            Get_FD_slope(coll, 'Close', 20, 3)
            Get_FD_slope(coll, 'Close', 20, 4)
#        else:
#            print 'Collection'+sys.argv[1]+' does not exist!'



