# Stock price prediction using sentiment polarity from news articles and deep learning

This is the first version and still work in progress.

Sentiment from news and deep learning (LSTM) is used to help predict stock prices. The functionality is exposed via Flask API and web page as a simple frontend. It is also possible to run it from console.

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

[fin-news-data-collection]: <https://github.com/andrejlukic/fin-news-data-collection>
[Loughran/McDonald finance-specific dictionary]: <https://sraf.nd.edu/>
[Customized financial lexicon]: <https://github.com/nunomroliveira/stock_market_lexicon>
[Machine Learning for Trading]: <https://github.com/ahmedhamdi96/ML4T>
[Scrapy]: <https://scrapy.org/>
