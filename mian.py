from newspaper import Article
import requests
from transformers import pipeline
#import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#BAD WORD = LOADING, ERROR, ADVERTISEMENT, Advertisement
import random
import os
import facebookdata
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
        clean_text = text.replace('\n', ' ')
        print(clean_text)
        print(len(clean_text))
        if len(clean_text) > 2500:
            AppropriateNews = True

    keywords = keywordsExtraction(clean_text)

    text = text_summary_small(clean_text)

    text = text + f"\n Source:{data['articles'][RandomURL]['url']}"
    for i in keywords:
        text = text + f" #{i}, "
    postToFacebook(text)



def postToFacebook(text):
    facebook = os.environ["facebookKey"]
    pageid = os.environ["pageId"]
    post_url = 'https://graph.facebook.com/{}/feed'.format(pageid)
    payload = {
    'message': text,
    'access_token': facebook
    }
    r = requests.post(post_url, data=payload)
    print(r.text)
    
def keywordsExtraction(text_):
    
    tokenizer = AutoTokenizer.from_pretrained("ilsilfverskiold/tech-keywords-extractor")
    model = AutoModelForSeq2SeqLM.from_pretrained("ilsilfverskiold/tech-keywords-extractor")

    
    inputs = tokenizer([text_], max_length=1024, return_tensors="pt")
    
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    result = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result.split(", ")

def text_summary_small(text_):
    model1 = 'google/pegasus-large'
    model2 = "facebook/bart-large-cnn"
    model3="sshleifer/distilbart-cnn-12-6"
    text = text_.replace('\n',"")
    pattern = r'(?<!\d)\d+\.\d+(?!\d)'
    text = re.sub(pattern, lambda match: match.group().replace('.', ','), text)

    text = text.replace('.', '.<eos>')
    text = text.replace('!', '!<eos>')
    text = text.replace('?', '?<eos>')
    sentences = text.split('<eos>')

    max_chunk = 500
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) +len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])

    summarizer = pipeline("summarization", model3)
    result = summarizer(chunks, max_length=200, min_length=50, do_sample=False)
    summary = ' '.join([summ["summary_text"] for summ in result])

    return summary


if __name__ == "__main__":
    main()