'''
Created on 23 Apr 2019

@author: Andrej Lukic
'''
import  pandas as pd
import numpy as np

def run_backtest(df):
    start_cash = 1000000
    infy_backtest = pd.DataFrame({"Start Port. Value": [],
                             "End Port. Value": [],
                             "End Date": [],
                             "Shares": [],
                             "Share Price": [],
                             "Trade Value": [],
                             "Profit per Share": [],
                             "Total Profit": [],
                             "Stop-Loss Triggered": []})
    port_value = 1  # Max proportion of portfolio bet on any trade
    batch = 10      # Number of shares bought per batch
    stoploss = .2    # % of trade loss that would trigger a stoploss
    last_action = None
    for index, row in df.iterrows():
        
        signal = row['Signal']
        
        if(signal == -1 and last_action is not None):           
            stocks_available = int(last_action['Shares'])
            trade_val =  row["Close"] * stocks_available        
            profit = row['Profit'] * stocks_available  
            #print('sell {} {} {} ->{}'.format(index, stocks_available, trade_val, last_action["End Port. Value"]))
            # Add a row to the backtest data frame containing the results of the trade
            last_action = {
                        "Start Port. Value": last_action["End Port. Value"],
                        "End Port. Value": last_action["End Port. Value"] + trade_val,
                        # "End Date": row["End Date"],
                        "End Date": index,
                        "Shares": 0,
                        "Share Price": row["Close"],
                        "Trade Value": trade_val,
                        "Profit per Share": row['Profit'],
                        "Total Profit": profit,
                        "Stop-Loss Triggered": False
                    }
            infy_backtest = infy_backtest.append(pd.DataFrame(last_action, index = [index]))        
        else:
            if(last_action is not None):
                cash = last_action['End Port. Value']
            else:
                cash = start_cash
            batches =  cash // np.ceil(batch * row["Close"]) # Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Close"] # How much money is put on the line with each trade
            shares = batches * batch
            #print('buy {} {} {} {} {}'.format(index, cash, trade_val, shares, cash - trade_val))
            if row["Low"] < (1 - stoploss) * row["Close"]:   # Account for the stop-loss
                share_profit = np.round((1 - stoploss) * row["Close"], 2)
                stop_trig = True
                print('stop triggered')
            else:            
                stop_trig = False
            
            share_profit = -row["Close"] 
            last_action = {
                        "Start Port. Value": cash,
                        "End Port. Value": cash - trade_val,
                        # "End Date": row["End Date"],
                        "End Date": index,
                        "Shares": shares,
                        "Share Price": row["Close"],
                        "Trade Value": trade_val,
                        "Profit per Share": share_profit,
                        "Total Profit": -trade_val,
                        "Stop-Loss Triggered": stop_trig
                    }
            infy_backtest = infy_backtest.append(pd.DataFrame(last_action, index = [index]))
    return infy_backtest