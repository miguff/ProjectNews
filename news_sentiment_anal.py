import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import LongformerModel, AutoModel, EncoderDecoderModel

import re
import sys
import torch



    
def text_summary_long(text_):
    nltk.download('stopwords')
    nltk.download('punkt')
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

# Creating a dictionary to keep the score
# of each sentence
    sentences = sent_tokenize(text_)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

# Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

# Storing sentences into our summary.
    summary = ''
    for sentence in sentences:

        if (sentence in sentenceValue) and (sentenceValue[sentence] > (average)):
            summary += " " + sentence

    return summary

def text_summary_small(text_, maxlen = None):
    model1 = 'google/pegasus-large'
    model2 = "facebook/bart-large-cnn"
    model3="sshleifer/distilbart-cnn-12-6"
    model4 = "pszemraj/led-base-book-summary"
    text = text_.replace('\n',"")
    pattern = r'(?<!\d)\d+\.\d+(?!\d)'
    text = re.sub(pattern, lambda match: match.group().replace('.', ','), text)

    summarizer = pipeline("summarization", model4, torch_dtype=torch.bfloat16)

    if maxlen == None:
        summary = summarizer(text,max_length = 512, do_sample=False)
    else:
        summary = summarizer(text, max_length=maxlen, do_sample=False)
    final_summary = summary[0]["summary_text"]

    return final_summary
