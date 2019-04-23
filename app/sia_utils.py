'''
Created on 23 Apr 2019

@author: Andrej Lukic
'''
'''
                    
                    Initialize sentiment analyzer
                    
'''
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

lex_path = r'.\lexicon_data\\'

def init_sia():
    #nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()    
    # load specialized stock market lexicon
    stock_lex = pd.read_csv(lex_path + 'stock_lex.csv')
    stock_lex['Sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
    stock_lex = dict(zip(stock_lex.Item, stock_lex.Sentiment))
    stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
    stock_lex_scaled = {}
    for k, v in stock_lex.items():
        if v > 0:
            stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
        else:
            stock_lex_scaled[k] = v / min(stock_lex.values()) * -4
    
    # Update with Loughran and McDonald list of keywords
    positive = []
    with open(lex_path + r'lm_positive.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            positive.append(row[0].strip())
        
    negative = []
    with open(lex_path + r'lm_negative.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            entry = row[0].strip().split(" ")
            if len(entry) > 1:
                negative.extend(entry)
            else:
                negative.append(entry[0])
    
    final_lex = {}
    final_lex.update({word:2.0 for word in positive})
    final_lex.update({word:-2.0 for word in negative})
    final_lex.update(stock_lex_scaled)
    final_lex.update(sia.lexicon)
    sia = SentimentIntensityAnalyzer()
    sia.lexicon = final_lex    
    return sia

def add_sentiment_polarity(sia, df):
    df['sentiment_vader_LM_title'] = pd.to_numeric(df['Title'].map(lambda a: sia.polarity_scores(a)['compound'])) 
    df['sentiment_vader_LM_body'] = pd.to_numeric(df['Body'].map(lambda a: sia.polarity_scores(a)['compound']))
    return df
    
def compute_signals(df):    

    sig_title = df.groupby(df.index).mean()
    #sig_title.sort_values(by='Date').head()
    
    sa_column = 'sentiment_vader_LM_body'
    
    sig_title["SA_Point"] = np.where(sig_title[sa_column] > 0.5, 1, sig_title[sa_column])
    sig_title["SA_Point"] = np.where(sig_title[sa_column] < -0.1, -1, sig_title["SA_Point"])
    sig_title["SA_Point"] = np.where(np.abs(sig_title["SA_Point"])!=1, 0, sig_title["SA_Point"])
    sig_title['Signal'] = pd.Series()
    sig_title['Profit'] = pd.Series()
    last_action = None
    for index, row in sig_title.iterrows():
        
        if(row["SA_Point"] == 0 or row["SA_Point"] == last_action):
            row['Signal'] = 0
        else:
            #print('{} last_action: {} new_action: {}'.format(index, last_action, row["SA_Point"]))
            row['Signal'] = last_action = row["SA_Point"]
            
    return sig_title