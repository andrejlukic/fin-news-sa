'''
Created on 23 Apr 2019

@author: Andrej Lukic
'''

import glob
import matplotlib.pyplot as plt 
import os
import pandas as pd
from pandas_datareader import data
import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
import hashlib
#from pandas.tests.io.parser import index_col

'''

                Load historical data
                
'''
stock_data_dir = r'./stocks/'
tiingo_api_key = os.environ['TiingoAPI']
stock_remote_src = 'Tiingo'

def load_stock(symbol, start_date=None, end_date=None):
    if(not start_date):
        start_date = datetime.date.today() - relativedelta(years=20)
        
    if(not end_date):
        end_date = datetime.date.today()
    print('Attempt to load {} data from {} to {}.'.format(symbol.upper(), start_date, end_date))
    df = get_local_cached_file(symbol, start_date, end_date)
    if(df is None):
        print('Update needed. Attempting remote fetch from {} ...'.format(stock_remote_src))
        df = data.DataReader(symbol.upper(), stock_remote_src.lower(), start_date, end_date, access_key=tiingo_api_key)
        df.reset_index(0, inplace=True)
        if(not df.empty):
            print('Ok. Received {} data points from {} to {}'.format(df.shape[0], df.index.min(), df.index.max()))
            save_to_cache(df, symbol, start_date, end_date)
            print('Saved to cache.')
        else:
            print('Received no data. Check that the stock symbol is correct.')
    else:
        print('Found cached version')
    return to_internal_format(df)
    

def get_local_cached_file(symbol, start_date, end_date):
    
    hash = hashlib.md5(symbol.encode() + str(start_date).encode()+ str(end_date).encode()).hexdigest()
    cached_file = Path(stock_data_dir + '{}_{}.csv'.format(symbol.upper(), hash))
                                    
    if not cached_file.is_file():   
        return None
    else:
        return pd.read_csv(cached_file, parse_dates=True)
def clean_up_old_cached_files(symbol):
    list_of_files = glob.glob('{}/{}*'.format(stock_data_dir, symbol.upper())) # * means all if need specific format then *.csv    
    for old_file in list_of_files:
        os.remove(old_file)

def save_to_cache(df, symbol, start_date, end_date):    
    #check if directory {stock_data_dir} exist
    if(not os.path.isdir(stock_data_dir)):
        os.mkdir(stock_data_dir)    
    hash = hashlib.md5(symbol.encode() + str(start_date).encode()+ str(end_date).encode()).hexdigest()    
    cached_path = stock_data_dir + '{}_{}.csv'.format(symbol.upper(), hash)
    clean_up_old_cached_files(symbol)                                
    df.to_csv(cached_path)
    
def debug_load_stock(stock_name, date_from=None, date_to=None):
    path_stock = stock_data_dir + stock_name.lower()+ "\\"    
    all_files = glob.glob(os.path.join(path_stock, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files))
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values(by='Date', ascending=True)
    df.set_index(df['Date'], inplace=True)
    
    if(date_from is not None):
        df = df[date_from:]
        
    if(date_to is not None):
        df = df[:date_to]
        
    #return debug_to_readable(df)

'''

                Display the stock
                
'''
def display_stock(df):
    plt.plot(df['Date'], df['Close'])
    plt.show()


'''

                Transform into more readable format
                
'''
def to_internal_format(df):
    try:    
        ohlc = df[['open', 'high', 'low','close', 'adjClose', 'volume']].copy()
        ohlc.index = pd.to_datetime(ohlc.index).date
        ohlc.columns = ['Open','High','Low','Close','Adj Close','Volume']
        ohlc.index.names = ['Date']
    except Exception as e:
        print(df.head())
        raise
    return ohlc    

#d = load_stock('infy')
#print(d.head(20))