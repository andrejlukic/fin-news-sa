# Stock price LSTM based on sentiment (draft)

Sentiment from news and deep learning (LSTM) is used to predict stock prices (NOTE: it does not work). The functionality is exposed via Flask API and web page as a simple frontend. It is also possible to run it from console.

# Usage

1. A stock symbol and company name are entered followed by pressing "Get forecasst"
    * a list of names separated by commas is also tolerated. The first word in the list will be used as a stock symbol and the rest as search terms for searching the articles for mentions of the company 
2. A 20 year history of the stock is downloaded and cached for further use
3. A LSTM RNN is trained using Keras and a Tensorflow backend on the stock history, a model is then saved for further use. Adapted from [Machine Learning for Trading].
4. News scraping engine is run using https://scrapy.org/ (separate repo - [fin-news-data-collection])
    * news from the front page are first parsed
    * each article URL is hashed and stored in a log so articles are downloaded only once
    * new articles are downloaded and stored in a csv
5. Articles for the past 6 days are filtered for the stock symbol
6. Sentiment polarity is calculated for the article titles. 
    * The algorithm used is Vader Sentiment Analyzer with a [customized financial lexicon] adapted to stock market conversations in microblogging services. The lexicon was further improved by the [Loughran/McDonald finance-specific dictionary]
7. The resulting stock price value prediction and sentiment polarity are displayed

# Modules

* **main** - main console app
* **bot_api** - using Flask API to exposing the function as a web service and SocketIO to update the frontend in a more user friendly way
* **ml_utils** - LSTM RNN implementaion, adapted from [Machine Learning for Trading]
* **sia_utils** - Vader Sentiment Analyzer implementaion
* **stock_utils** - download, caching and preparation of stock history data-collection
* **article_utils** - caching and preparation of news articles
* **spider_utils** - scraping of news articles (using Scrapy and scraper defined in [fin-news-data-collection])
* **lexicon_data/** - directory containing the [customized financial lexicon] and the [Loughran/McDonald finance-specific dictionary]
* **templates/** - directory containing the frontend page in HTML format
* **static/** - directory containing the resources for the frontend page in HTML format

# Installation (dev env)

Python 3.6 and the following libraries are required: *pandas scrapy regex matplotlib pandas_datareader nltk keras tensorflow sklearn flask_socketio*

(Google Cloud g1-small (1 vCPU, 1.7 GB memory) machine is enough)

1. Create a new directory and get the code
    * [fin-news-sa] for the main  app
    * [fin-news-data-collection] for the scraper
2. Add the scraper project under the Python path (example Ubuntu 18.04.1 LTS):
    * SITEDIR=$(python3 -m site --user-site)
    * mkdir -p "$SITEDIR" (if needed)
    * echo "$HOME/dir-where-fin-news-data-collection-is" > "$SITEDIR/indianstock.pth"
3. Configure the Tiingo API key and set the env var (example Ubuntu 18.04.1 LTS):
    * register on [Tiingo] and get the API key
    * export TiingoAPI="aaaa123456789bbbbbbbbbbbbbbbbbbbbbb"
4. Configure the connection between the web page and web service:
    * edit templates/try-bootstrap.html and change the URL in two places
    * under AJAX request (search for "url :" or "$(document).ready(function() {")
    * under SocketIO connect (search for "var socket = io.connect(")
5. Run the Flask backend - "python3 bot-api.py" and hit the URL, you should see a simple dashboard open. 

[fin-news-data-collection]: <https://github.com/andrejlukic/fin-news-data-collection>
[fin-news-sa]: <https://github.com/andrejlukic/fin-news-sa>
[Loughran/McDonald finance-specific dictionary]: <https://sraf.nd.edu/>
[Customized financial lexicon]: <https://github.com/nunomroliveira/stock_market_lexicon>
[Machine Learning for Trading]: <https://github.com/ahmedhamdi96/ML4T>
[Scrapy]: <https://scrapy.org/>
[Tiingo]: <https://www.tiingo.com>
