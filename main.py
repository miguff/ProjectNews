from newspaper import Article
import requests
from transformers import pipeline
#import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re
import sys

#BAD WORD = LOADING, ERROR, ADVERTISEMENT, Advertisement
import random
import os
import facebookdata
from news_sentiment_anal import text_summary_small, text_summary_long, keywordsExtraction
from wordpress_upload import postToFacebook, wordpress


#NBC news - nbc-news
#BBC = bbc-news
#MSNBC = msnbc
#Fox News - fox-news
#Fortune - fortune ?
#Business Insider - business-insider
#Australian Financial Review - australian-financial-review
#AP - associated-press
#ABC - abc-news
#The Huffington Post - the-huffington-post
#The Irish Times = the-irish-times
#The Times of india - the-times-of-india
#The Washington Post - the-washington-post
#USA today - usa-today

def main():

    AppropriateNews = False
    while AppropriateNews == False:
        RandomNews = random.randint(0, 12)
        RandomURL = random.randint(0, 4)
        print(RandomNews)
    
        NewsPaperDict = {
        0:"nbc-news",
        1:"bbc-news",
        2:"msnbc",
        3:"fox-news",
        4:"business-insider",
        5:"australian-financial-review",
        6:"associated-press",
        7:"abc-news",
        8:"the-huffington-post",
        9:"the-irish-times",
        10:"the-times-of-india",
        11:"the-washington-post",
        12:"usa-today"

    }
        x  = requests.get(f"https://newsapi.org/v2/top-headlines?sources={NewsPaperDict[RandomNews]}&apiKey=e9a76c1252144c2d9609b3bf837bc375")
        data = x.json()

        url = data['articles'][RandomURL]['url']
        article = Article(url)

        article.download()
        article.parse()

        text = article.text
        Title = article.title
        clean_text = text.replace('\n', ' ')
        print(clean_text)
        print(len(clean_text))
        if len(clean_text) > 2000:
            AppropriateNews = True

    keywords = keywordsExtraction(clean_text)

    
    textwp = text_summary_long(clean_text)
    text = text_summary_small(clean_text)

    print(textwp)
    print('---------------------')


    wp_url = wordpress(title = Title, body=textwp, cikk_url=data['articles'][RandomURL]['url'], version="V1")
    text = text + f"\n Source:{wp_url}"
    for i in keywords:
        text = text + f" #{i}, "
    
    postToFacebook(text)



    




if __name__ == "__main__":
    main()