'''
Created on 23 Apr 2019

@author: Andrej Lukic
'''
import spider_utils as su
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import csv
import re

path_to_scraper = '.'
path_to_news_history = r'.\history\\'

def load_news(stock_symbol = None, sources = None, start_date = None, end_date = None):
    df = load_news2(stock_symbol, sources, start_date, end_date)
    #df = clean_up(df)
    return df
    
def load_news2(stock_symbol = None, sources = None, start_date = None, end_date = None, depth=0):
    
    if(not start_date):
        start_date = datetime.date.today() - relativedelta(days=5)

    if(not end_date or end_date > datetime.date.today()):
        end_date = datetime.date.today()
           
    downloaded = path_to_scraper+'\\'+su.get_spider_outputpath()
    df = pd.DataFrame()
    if(os.path.isfile(downloaded)):
        print('Loading saved articles from {} ... '.format(downloaded))
        df = read_local_articles(downloaded)
        if(not df.empty):
            last_date = df.index.max()
            min_date = df.index.min()
            print('Loaded articles from {0} to {1} ... '.format(min_date, last_date))
            if(last_date >= end_date and min_date < start_date):      
                print('Slicing date range {0}-{1} ... '.format(start_date, end_date))
                return df.loc[start_date:end_date]
            else:
                print('Previously saved articles span date range {0}-{1} ... '.format(start_date, end_date))
    if(depth > 0):
        print('Warning: articles are not covering complete date range')
        return df
    else:   
        print('Info: try to get more fresh content ... ')     
        su.run_spiders(sources)
        load_news2(stock_symbol = stock_symbol, start_date = start_date, end_date = end_date, depth = depth+1)
    
def filter_by_stock(df, term_list):
    if(not isinstance(df, pd.DataFrame) or df.empty):
        return None
    if(not term_list):
        return df    
    s = re.compile('|'.join(term_list), flags=re.IGNORECASE)
    return df[df['Body'].str.contains(s) | df['Title'].str.contains(s)]
                                      
def read_local_articles(path):
    # stock market lexicon
    df = pd.read_csv(path, delimiter=';') 
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['body'] = df['body'].astype(str)  
    df.sort_values(by='date', inplace=True)
    df = df.rename(str.capitalize,axis='columns')
    df.set_index('Date', inplace=True)       
    return df

def load_news_history(stock_symbol):
    # stock market lexicon
    # stock market lexicon
    path = '{0}old-news-{1}.csv'.format(path_to_news_history,stock_symbol)
    if(not os.path.isfile(path)):
        print('Historical data for {0} not available.'.format(stock_symbol))
        
    df = pd.read_csv(path, delimiter=';') 
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['body'] = df['body'].astype(str)  
    df.sort_values(by='date', inplace=True)
    df = df.rename(str.capitalize,axis='columns')
    df.set_index('Date', inplace=True)       
    return df

def clean_up(df):
    return df.groupby('Url').max()
