'''
Created on 23.11.2018

@author: al
'''
import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug import secure_filename
import numpy as np
import pandas as pd
from main import current_state
from keras.backend import clear_session
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/", methods=['GET'])
def frontpage():    
    return render_template('try-bootstrap.html')

@app.route("/getforecast", methods=['POST'])
def index():    
    #ticker = request.form.get('ticker').strip()
    search_terms = request.form.get('search_terms')
    #if(ticker and search_terms):
    if(search_terms):        
        search_terms = search_terms.split(',')
        search_terms = [s.strip() for s in search_terms]
        #to work with the bootstrap one input field frontend:
        ticker = search_terms[0]
        if(len(search_terms)>1):
            search_terms = search_terms[1:]        
        print('Debug-Form values: {}-{}'.format(ticker,search_terms))
        #To resolve Keras error with multiple threads, clear session: (TypeError: Cannot interpret feed_dict key as Tensor:)
        clear_session()
        forecast, sentiment, hist6M = current_state(ticker, search_terms, socketio)
        return jsonify({'output':'Forecast: {0:.2f}$ - sentiment: {1:.2f}'.format(forecast, sentiment),
                        'hist6Mval': hist6M.tolist(), 'hist6Mlab': hist6M.index.strftime('%x').tolist()})
    else:
        return jsonify({'error' : 'Please enter stock symbol and search terms!'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0',port=80, debug=True)
    app.run(host='0.0.0.0',port=80,debug=True)