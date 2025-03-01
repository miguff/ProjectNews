from newspaper import Article
from newsapi import NewsApiClient
import facebookdata
import translators as ts

import requests
from transformers import pipeline
#import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re
import sys

#BAD WORD = LOADING, ERROR, ADVERTISEMENT, Advertisement
import random
import os

from news_sentiment_anal import text_summary_small, text_summary_long 
from wordpress_upload import postToFacebook, wordpress
import sys


random.seed(8)

def main():

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-hu")
    

    newsapi = NewsApiClient(api_key=os.environ["newsapi"])

    AppropriateNews = False
    while AppropriateNews == False:

        all_articles = newsapi.get_everything(q='Hungary',
                                      language='en',
                                      sort_by='relevancy')

        RandomURL = random.randint(0, len(all_articles['articles']))

        url = all_articles['articles'][RandomURL]['url']
        article = Article(url)

        article.download()
        article.parse()

        text = article.text
        Title = article.title
        clean_text = text.replace('\n', ' ')
        if len(clean_text) > 2000:
            AppropriateNews = True


    #keywords = keywordsExtraction(clean_text)
    #print(clean_text)
    #print(keywords)



    #textwp = text_summary_small(clean_text)

    textwp = "Sign up for the Slatest to get the most insightful analysis, criticism, and advice out there, delivered to your inbox daily. In March, Trump sued ABC, CBS, and host George Stephanopoulos over comments he made during a March 10 interview with South Carolina Rep. Nancy Mace in which Stephanopoulos said that Trump had been found liable for rape. The jury found Trump civilly liable for sexually abusing E. Jean Carroll , but the jury did not find for her on that claim. Trump's lawsuit against ABC is likely to meet the same fate as his other defamation actions against the media: failure based on the public record. News organizations are also obeying an increasingly authoritarian Trump administration. For example, Facebook parent company Meta has given $1 million to Trump's inauguration fund. Major players in old and new media are following suit. Take Meta founder Mark Zuckerberg, for example. He once championed democracy and lambasted foreign efforts to interfere in American elections. But since the election, Zuckerberg has gone to Mar-a-Lago to dine with Trump. And finally, big tech companies are cozying up to Trump. They threaten legal action against media outlets if they don't back down."

    print(textwp)    
    splitted_text = textwp.split(".")
    print(splitted_text)

    translatedtext = []
    for i in splitted_text:
        traslate = ts.translate_text(i, translator="google", to_language = "hu")
        translatedtext.append(traslate)

    print(translatedtext)

    print('---------------------')
    sys.exit()


    #textwp = text_summary_long(clean_text)
    text = text_summary_small(textwp, 120)
    print(text)
    print()
    print('---------------------')
    sys.exit()

    textlonghu = ""
    textsmallhu = ""
    Title = ""
    

    #textlonghu = ts.translate_text(textwp, translator="google", to_language = "hu")
    #textsmallhu = ts.translate_text(text, translator="bing", to_language = "hu")
    #Title = ts.translate_text(Title, translator="bing", to_language="hu")

    print(textlonghu)
    #print(textsmallhu)
    #print(Title)
    sys.exit()
    
    wp_url = wordpress(title = Title, body=textlonghu, cikk_url=url, version="V1", excerpt=textsmallhu)
    sys.exit()
    text = text + f"\n Source:{wp_url}"
    for i in keywords:
        text = text + f" #{i}, "
    
    postToFacebook(text)


def split_sentence(text):

    sentence_spl_regex = (
        r'(?<!\b[A-Z]\.)(?<!\b[A-Z]\.[A-Z]\.)(?<!\b(?:Mr|Mrs|Dr|St|E|PhD|Prof)\.)'
        r'(?<=[.!?])\s+(?=[A-Z])'
    )

    sentenes = re.split(sentence_spl_regex, text)
    return [sentence.strip() for sentence in sentenes if sentence.strip()]


    




if __name__ == "__main__":
    main()