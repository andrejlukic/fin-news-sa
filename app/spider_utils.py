'''
Created on 23 Apr 2019

@author: Andrej Lukic
'''
import pandas as pd
import os
from subprocess import call
from twisted.internet import reactor
from scrapy.crawler import Crawler
from scrapy import log, signals
from scrapy.settings import Settings
from scrapy.utils.project import get_project_settings
from indianstock.spiders.up_marscr import MarScreSpider
from indianstock.spiders.up_reu import ReutersSpider
from indianstock.spiders.up_bussta import BusStaSpider
from indianstock.spiders.up_itimes import ItimesSpider
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

scraper_project = 'indianstock'
# supported spiders: reuters, itimes, bussta, marscre

def initspider(sname_list):    
    if(not sname_list):
        return [MarScreSpider()]
    else:
        slist = []
        for sn in sname_list:
            if(sn == 'reuters'):
                slist.append(ReutersSpider())
            elif(sn == 'itimes'):
                slist.append(ItimesSpider())
            elif(sn == 'bussta'):
                slist.append(BusStaSpider())
            elif(sn == 'marscre'):
                slist.append(MarScreSpider())
        return slist
        
def run_spiders(slist = None):
    
    spider_list = initspider(slist)
    
    settings = Settings()
    os.environ['SCRAPY_SETTINGS_MODULE'] ='{}.settings'.format(scraper_project)
    settings_module_path = os.environ['SCRAPY_SETTINGS_MODULE']
    settings.setmodule(settings_module_path, priority='project')
    
    configure_logging()
    runner = CrawlerRunner(settings=settings)
    
    @defer.inlineCallbacks
    def crawl():
        for spider in spider_list:
            print('Starting spider {}'.format(type(spider)))
            yield runner.crawl(spider)        
        reactor.stop()
    
    crawl()
    reactor.run() # the script will block here until the last crawl call is finished

def get_spider_settings():
    settings = Settings()
    os.environ['SCRAPY_SETTINGS_MODULE'] ='{0}.settings'.format(scraper_project)
    settings_module_path = os.environ['SCRAPY_SETTINGS_MODULE']
    settings.setmodule(settings_module_path, priority='project')
    return settings

def get_spider_outputpath():
    s = get_spider_settings()
    return s['FEED_URI']

# run_spiders()
# s = get_spider_outputpath()
# print(s)