
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def analize(text):
    nltk.download([
    "stopwords",
    'vader_lexicon',
    'punkt'])
    sia = SentimentIntensityAnalyzer()
    translated_text = text

    return is_positive(sia, translated_text)


def is_positive(sia, text: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    print(text)
    score = sia.polarity_scores(text)["compound"]
    print(sia.polarity_scores(text))

    if score > 0:
        return "Ez a cikk pozitívan vetíti előre a gazdaság alakulását"
    else:
        return "Ez egy negatív cikk"