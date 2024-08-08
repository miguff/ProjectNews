import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import sys



    
def text_summary_long(text_):
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

    summarizer = pipeline("summarization", model1)
    result = summarizer(chunks, max_length=60, min_length=30, do_sample=False)
    summary = ' '.join([summ["summary_text"] for summ in result])
    return summary


def keywordsExtraction(text_):
    
    tokenizer = AutoTokenizer.from_pretrained("ilsilfverskiold/tech-keywords-extractor")
    model = AutoModelForSeq2SeqLM.from_pretrained("ilsilfverskiold/tech-keywords-extractor")

    
    inputs = tokenizer([text_], max_length=1024, return_tensors="pt")
    
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    result = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result.split(", ")