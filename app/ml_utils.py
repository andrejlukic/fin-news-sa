'''
Created on 23 Apr 2019

@author: Andrej Lukic

Thanks to ahmedhamdi96, who shared his code on GitHub and is the base for 95% of the following code. Please find his original at: https://github.com/ahmedhamdi96/ML4T

'''
from keras.callbacks import EarlyStopping
import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import os
import glob
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

stocks_historical_data_dir = './stocks'


def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(Y_test, predictions, Y_test_inv_scaled, predictions_inv_scaled):
    rmse = (mean_squared_error(Y_test, predictions) ** 0.5)
    print('\nNormalized RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test, predictions) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test, predictions)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test, predictions)
    print('Normalized MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test.T, predictions.T)
    print("Normalized Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test, predictions)
    print("Normalized r^2: %.3f"%(r2))
    normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    #evaluating the model on the inverse-normalized dataset
    rmse = (mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5)
    print('\nInverse-Normalized Outsample RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test_inv_scaled, predictions_inv_scaled)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test_inv_scaled, predictions_inv_scaled)
    print('Inverse-Normalized Outsample MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test_inv_scaled.T, predictions_inv_scaled.T)
    print("Inverse-Normalized Outsample Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test_inv_scaled, predictions_inv_scaled)
    print("Inverse-Normalized Outsample r^2: %.3f"%(r2))
    inv_normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    return normalized_metrics, inv_normalized_metrics

def compute_lag_metric(actual, prediction, lookup, symbol):
    diff_list = [None] * lookup
    lag_list = [None] * (len(actual)-lookup+1)

    for i in range(len(actual)-lookup+1):
        for j in range(lookup):
            diff_list[j] = abs(actual[i] - prediction[i+j])
        lag_list[i] = diff_list.index(min(diff_list))

    max_diff_count = [0] * lookup

    for i in range(len(lag_list)):
        max_diff_count[lag_list[i]] += 1

    _, ax = plt.subplots()
    ax.bar(range(len(max_diff_count)), max_diff_count, align='center')
    plt.sca(ax)
    plt.title(symbol+" Lag Test")
    ax.set_xlabel('Day Lag')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    _, ax1 = plt.subplots()
    index = actual[:len(actual)-lookup+1].index
    ax1.scatter(index, lag_list)
    plt.title(symbol+" Daily Lag Test")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Lag')
    ax1.grid(True)

    return lag_list

def symbol_to_path(symbol):
    #get latest file that starts with stock symbol name:
    list_of_files = glob.glob('{}/{}*'.format(stocks_historical_data_dir, symbol.upper())) # * means all if need specific format then *.csv
    return max(list_of_files, key=os.path.getctime)

def get_stock_data(symbol, start_date=None, end_date=None, columns=["Date", "Adj Close"]):

    df = pd.read_csv(symbol_to_path(symbol), index_col="date",
                     parse_dates=True, usecols=columns,
                     na_values="nan")
    print('Read stock data for {0} records: {1} (using: {2})'.format(symbol, df.shape[0], symbol_to_path(symbol)))
    df.index.names = ['Date']
    df.columns = ['Adj Close','Volume']
    print('Data min {} Data max {} Start {} End {}'.format(df.index.min(),df.index.max(), start_date, end_date))
    return df[start_date:end_date]

#BEGIN
def compute_momentum_ratio(prices, window):
    #first window elements >> NA
    momentum_ratio = (prices/prices.shift(periods = 1)) - 1
    return momentum_ratio

def compute_sma_ratio(prices, window):
    #Simple Moving Average
    #first window-1 elements >> NA
    sma_ratio = (prices / prices.rolling(window = window).mean()) - 1
    return sma_ratio

def compute_bollinger_bands_ratio(prices, window):
    #first window-1 elements >> NA
    bb_ratio = prices - prices.rolling(window = window).mean()
    bb_ratio = bb_ratio / (2 * prices.rolling(window = window).std())
    return bb_ratio

def compute_volatility_ratio(prices, window):
    #first window-1 elements >> NA
    volatility_ratio = ((prices/prices.shift(periods = 1)) - 1).rolling(window = window).std()
    return volatility_ratio

def compute_vroc_ratio(volume, window):
    #Volume Rate of Change
    #first window-1 elements >> NA
    vroc_ratio = (volume/volume.shift(periods = window)) - 1
    return vroc_ratio
#END

def bulid_TIs_dataset(stock_symbol, start_date, end_date, window, normalize=True):    
    cols = ["date", "adjClose", "volume"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    df['momentum'] = compute_momentum_ratio(df['price'], window)
    df['sma'] = compute_sma_ratio(df['price'], window)
    df['bolinger_band'] = compute_bollinger_bands_ratio(df['price'], window)
    df['volatility'] = compute_volatility_ratio(df['price'], window)
    df['vroc'] = compute_vroc_ratio(df['Volume'], window)
    df['actual_price'] = df['price']
    df.drop(columns=["Volume"], inplace=True)
    df = df[window:]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
        df['momentum'] = scaler.fit_transform(df['momentum'].values.reshape(-1,1))
        df['sma'] = scaler.fit_transform(df['sma'].values.reshape(-1,1))
        df['bolinger_band'] = scaler.fit_transform(df['bolinger_band'].values.reshape(-1,1))
        df['volatility'] = scaler.fit_transform(df['volatility'].values.reshape(-1,1))
        df['vroc'] = scaler.fit_transform(df['vroc'].values.reshape(-1,1))
        df['actual_price'] = scaler.fit_transform(df['actual_price'].values.reshape(-1,1))
        
    #print(df.head())
    #print(df.tail())
    return df, scaler

def lstm_dataset_reshape(dataset, time_steps, future_gap, split):
    # ['price', 'momentum', 'sma', 'bolinger_band', 'volatility', 'vroc', 'actual_price']
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    #print("X Shape:", X.shape)
    #print("Y Shape:", Y.shape)

    X_sampled = []
    for i in range(X.shape[0] - time_steps + 1):
        X_sampled.append(X[i : i+time_steps])
    X_sampled = np.array(X_sampled)
    #print("Sampled X Shape:", X_sampled.shape)
    future_gap_index = future_gap - 1
    X_sampled = X_sampled[:-future_gap]
    Y_sampled = Y[time_steps+future_gap_index: ]
    #print("Applying Future Gap...")
    #print("Sampled X Shape:", X_sampled.shape)
    #print("Sampled Y Shape:", Y_sampled.shape)

    if split != None:
        split_index = int(split*X_sampled.shape[0])
        X_train = X_sampled[:split_index]
        X_test = X_sampled[split_index:]
        Y_train = Y_sampled[:split_index]
        Y_test = Y_sampled[split_index:]
        print("(X_train, Y_train, X_test, Y_test) Shapes:")
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    return X_sampled, Y_sampled

def build_model(time_steps, features, neurons, drop_out, decay=0.0):
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(time_steps, features), return_sequences=True))
    model.add(Dropout(drop_out))
        
    model.add(LSTM(neurons[1], input_shape=(time_steps, features), return_sequences=False))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2], activation='relu'))        
    model.add(Dense(neurons[3], activation='linear'))

    adam = Adam(decay=decay)
    model.compile(loss='mse',optimizer=adam)
    model.summary()
    return model

def model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks):

    history = model.fit(
    X_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = validation_split,
    verbose = verbose,
    callbacks = callbacks
    )

    return history

def test_prediction(stock_symbol, start_date, end_date, window, future_gap, time_steps,
              neurons, drop_out, batch_size, epochs, validation_split, verbose, callbacks):
    #building the dataset
    #print("> building the dataset...")
    df_train, _ = bulid_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for LSTM
    #print("\n> reshaping the dataset for LSTM...")
    #print(df_train.columns)
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = lstm_dataset_reshape(ds_train, time_steps, future_gap, None)
    X_test, Y_test = lstm_dataset_reshape(ds_test, time_steps, future_gap, None)
    #building the LSTM model
    #print("\n> building the LSTM model...")
    features = X_train.shape[2]
    
    if(os.path.isfile('{}.h5'.format(stock_symbol))):
        print('\n> Loading cached model')      
        model = keras.models.load_model('{}.h5'.format(stock_symbol))
    else:
        print("\n> building the LSTM model...")
        model = build_model(time_steps, features, neurons, drop_out)
        #fitting the training data
        print("\n> fitting the training data...")
        model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks)
        print('\n> saving model')
        model.save('{}.h5'.format(stock_symbol))
    #predictions
    #print("\n> testing the model for predictions...")
    # last data point:
    #X_test = np.array([[[0.99162781, 0.46983332, 0.48282569, 1., 0.07956116, 0.25777586]]])
    predictions = model.predict(X_test)
    print('predictions: {}'.format(predictions))
    #inverse-scaling
    #print("\n> inverse-scaling the scaled values...")
    predictions = predictions.reshape((predictions.shape[0], 1))
    predictions_inv_scaled = scaler.inverse_transform(predictions)
    #print(predictions)
    #return predictions_inv_scaled
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    Y_test_inv_scaled = scaler.inverse_transform(Y_test)
    #evaluation
    normalized_metrics, inv_normalized_metrics = evaluate(Y_test, predictions, 
                                                          Y_test_inv_scaled, predictions_inv_scaled)
    #grouping the actual prices and predictions
    #print("\n> grouping the actual prices and predictions...")
    feature_cols = df_test.columns.tolist()
    feature_cols.remove("actual_price")
    df_test.drop(columns=feature_cols, inplace=True)
    df_test.rename(columns={"actual_price" : 'Actual'}, inplace=True)
    df_test = df_test.iloc[time_steps+future_gap-1:]
    df_test['Actual'] = Y_test_inv_scaled
    df_test['Prediction'] = predictions_inv_scaled

    return normalized_metrics, inv_normalized_metrics, df_test

def forecast(stock_symbol, num_steps = 1, test_start_date = None, test_end_date = None):
    if(not test_start_date):
        test_start_date = datetime.date.today() - relativedelta(days=10)
        
    if(not test_end_date):
        test_end_date = datetime.date.today() - relativedelta(days=1)
    
    print('Set date range {} - {}'.format(test_start_date, test_end_date))
    
    #LSTM and LinReg PAL
    dates_dic = {
        stock_symbol  : [test_start_date, test_end_date]
    }
    metrics_dic = {
        'LSTM'   : []
    }
    
    window = 2
    future_gap = 1
    time_steps = 1
    neurons = [256, 256, 32, 1]
    drop_out = 0.2                                   
    batch_size = 2048
    epochs = 300
    validation_split = 0.1
    verbose = 1
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                            patience=50, verbose=verbose, mode='auto')
    callbacks = [early_stopping_callback] 
    start_date = dates_dic[stock_symbol][0]
    end_date = dates_dic[stock_symbol][1]
    #print('Set date range {} - {}'.format(start_date, end_date))
    #LSTM
    normalized_metrics, inv_normalized_metrics, df = test_prediction(stock_symbol, start_date, 
    end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, validation_split, 
    verbose, callbacks)
    metrics_dic['LSTM'] = normalized_metrics
    #PAL
    lookup = 5
    #lag_list = compute_lag_metric(df['Actual'], df['Prediction'], lookup, stock_symbol)
    #df = df[:len(df)-lookup+1]
    #Price Forecast Plot    
    return metrics_dic, df, df[:len(df)-lookup+1]

#stock = 'INFY'
#metrics_dic, df = train_lstm(stock, '2018-06-01', '2019-06-01')
#print(metrics_dic)
#plot_data(df, stock+" Forecast (LSTM)", "Date", "Price", show_plot=False)
#plt.show()