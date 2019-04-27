'''
Created on 1 Apr 2019

@author: Andrej
'''
import pandas as pd
from subprocess import call
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from fndc.indianstock.spiders.up_marscr import MarScreSpider
from scrapy.settings import Settings
from scrapy.utils.project import get_project_settings
import os
import glob
import matplotlib.pyplot as plt 
from stock_utils import *
import sia_utils as su
import article_utils as au
import datetime as dt
import numpy as np
import backtesting as bt
import ml_utils as mlu
from flask_socketio import SocketIO

stock_symbol = 'infy'
stock_search_terms = ['infy','infosys']

def out(socketio, msg):
    print(msg)
    if(socketio is not None):
        socketio.emit('debugmsg', msg)

# def full_history_analysis(stock_symbol, stock_search_terms):
#     #start = dt.date(2014,4,20)
#     #end = dt.date(2017,6,27)
#             
#     '''    
#             Load INFY stock                    
#     '''    
#     print('1. Preparing {} stock data'.format(stock_symbol))
#     ohlc = load_stock(stock_symbol, None, None)
#     print('... received {2} data points from {0} to {1}'.format(ohlc.index.min(), ohlc.index.max(), stock_symbol))
#     plt.plot(ohlc.index, ohlc['Close'])
#     plt.show()
#     
#     '''    
#             Load articles for stock and calculate polarity for article title and for article body                
#     '''    
#     print('2. Loading all previously collected news articles.')
#     articles = au.load_news_history(stock_symbol)
#     print('... found {} articles'.format(articles.shape[0]))
#     print('3. Filtering news for keywords: [{0}]'.format(','.join(stock_search_terms)))
#     articles = au.filter_by_stock(articles, stock_search_terms)
#     print('... matched {0} ranging from {2}  to {3}'.format(articles.shape[0],stock_symbol, articles.index.min(), articles.index.max())) 
#     
#     '''                        
#             Initialize sentiment analyzer                        
#     '''
#     sia = su.init_sia()
#     print('4. Compute sentiment polarity for selected articles.')
#     articles = su.add_sentiment_polarity(sia, articles)
#     print('... done. {0}'.format(articles.shape))
#         
#     '''        
#             Sentiment: calculate average sentiment of the day
#                calculate simple signals based on sentiment
#                 0 = no action
#                -1 = sell
#                +1 = buy    
#             Price - Take closing price from where Signal is BUY
#             Profit column - a series of price differences (today - yesterday closing price)
#             End Date      - take date of the price of the day for which day before was Buy and Regime=1    
#     '''
#     print('5. Backtesting preparation, compute buy/sell signals, then merge with stock dataset')
#     articles_signals = su.compute_signals(articles)
#     print('... ok signals computed {0}'.format(articles_signals.shape))
#     articles_signals = articles_signals.join(ohlc[['Close','Low']], how='inner')
#     #articles_signals = articles_signals[start:end]   
#     articles_signals = articles_signals[articles_signals['Signal'] != 0]
#     articles_signals['Profit'] = articles_signals['Close'] - articles_signals['Close'].shift(1)
#     articles_signals = articles_signals[1:]
#     print('... ok datasets merged {0}'.format(articles_signals.shape)) 
#  
#     '''
#         Backtesting general        
#     '''
#     print('6. Run backt on the dataset.')
#     res = bt.run_backtest(articles_signals)
#     print('... done. Resulting dataset: {}'.format(res.shape[0]))
#     print(res)
#     res.to_excel('test_result.xlsx')

def current_state(stock_symbol, stock_search_terms, socketio):        
    #start = datetime.date.today() - relativedelta(years=20)
    #end = datetime.date.today() - relativedelta(days=2)
            
    '''    
            Load INFY stock                    
    '''    
    out(socketio,'1. Connecting to Tiingo API to download {} historical data:'.format(stock_symbol))
    ohlc = load_stock(stock_symbol, None, None)
    numdp = 0
    if(isinstance(ohlc, pd.DataFrame) and not ohlc.empty):
        numdp = ohlc.shape[0]
    out(socketio,'... received {3} records for {2}. Date range: {0} - {1}'.format(ohlc.index.min(), ohlc.index.max(), stock_symbol, numdp))
    #plt.plot(ohlc.index, ohlc['Close'])
    #plt.show()
    
    '''    
            Load articles for stock and calculate polarity for article title and for article body                
    '''    
    out(socketio,'2. Running news site scraper to collect fresh news articles:')
    try:
        articles = au.load_news(stock_symbol)
    except twisted.internet.error.ReactorNotRestartable as e:  # as e syntax added in ~python2.5
        out(socketio,'Error: an older scraping process is still executing. This might take a few minutes. Possible solution: restart the scraper. {}'.format(e))
        raise
    except:       
        raise
    sentiment = -100
    if(isinstance(articles, pd.DataFrame) and not articles.empty):
        out(socketio,'... found {} articles'.format(articles.shape[0]))
        out(socketio,'3. Filtering news articles for selected keywords: [{0}]:'.format(','.join(stock_search_terms)))
        articles = au.filter_by_stock(articles, stock_search_terms)    
        out(socketio,'... matched {0} articles. Date range {2} - {3}'.format(articles.shape[0],stock_symbol, articles.index.min(), articles.index.max()))
    else:
        out(socketio,'No fresh articles found.') 
    
    if(isinstance(articles, pd.DataFrame) and not articles.empty):
        
        '''                        
                Initialize sentiment analyzer                        
        '''
        sia = su.init_sia()
        out(socketio,'4. Compute sentiment polarity for selected articles.')
        articles = su.add_sentiment_polarity(sia, articles)
        articles["roll_sent"]=articles['sentiment_vader_LM_title'].rolling(3).mean()
        print(articles[['Title','sentiment_vader_LM_title']].tail(20))
        sentiment=articles["roll_sent"].iloc[-1]
        out(socketio,'... done.')
        '''        
                Sentiment: calculate average sentiment of the day
                   calculate simple signals based on sentiment
                    0 = no action
                   -1 = sell
                   +1 = buy    
                Price - Take closing price from where Signal is BUY
                Profit column - a series of price differences (today - yesterday closing price)
                End Date      - take date of the price of the day for which day before was Buy and Regime=1    
        '''
        #out(socketio,'5. Backtesting preparation, compute buy/sell signals, then merge with stock dataset')
        #articles_signals = su.compute_signals(articles)
        #out(socketio,'... ok signals computed')
        #articles_signals = articles_signals.join(ohlc[['Close','Low']], how='inner')           
        #articles_signals = articles_signals[articles_signals['Signal'] != 0]
        #articles_signals['Profit'] = articles_signals['Close'] - articles_signals['Close'].shift(1)        
        #out(socketio,'... ok datasets merged') 
    else:
        out(socketio,'... skipping sentiment analysis, because no articles that matched selected stock symbol keywords.')
 
    out(socketio,'5. Train and apply LSTM:')
    next = mlu.forecast(stock_symbol, 1)  
    out(socketio,'Forecast: {0:.2f}$'.format(round(next,2))) 
    
    #prepare data for the 6M plot:
    fromdate = datetime.date.today() - relativedelta(days=180)
    hist6M = ohlc[fromdate:]['Adj Close']
    
    return next, sentiment, hist6M

if __name__ == '__main__':
    #f,s = current_state(stock_symbol, stock_search_terms, None)
    #print("Sentiment={}".format(s))
    #s=sia.init_sia()
    #print('Forecast: {0:.2f}$'.format(f))    
    ohlc = load_stock(stock_symbol, None, None)
    print(ohlc.head(5))
    
    #dti = pd.to_datetime('2019-01-01')
    print(isinstance(ohlc.index, pd.DatetimeIndex))
    #filter_mask = ohlc.index > dti.date
    fromdate = datetime.date.today() - relativedelta(days=180)
    hist3M = ohlc[fromdate:]
    print(hist3M['Adj Close'].tolist())


