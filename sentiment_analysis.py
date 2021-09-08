import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
nltk.download('stopwords')

stop_Words = stopwords.words('english')
stop_Words.pop(0)

def vader_cleaned_score(file):
    sentiment_score = 0
    count = 0
    with open(file) as input:
        text = input.read()
        for word in stop_Words:
            text = re.sub(word,'',text)
        text = re.sub(r"[,\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        text = text.split('. ')
        for i in range(len(text)):
            text[i] = text[i].strip().replace('\n',' ')
            sentiment_score += analyzer.polarity_scores(text[i])['compound']
            count+=1
    return(sentiment_score/count) 

def vader_uncleaned_score(file):
    sentiment_score = 0
    count = 0
    with open(file) as input:
        text = input.readlines()
        for i in text:
            i = i.strip().replace('\n',' ')
            sentiment_score += analyzer.polarity_scores(i)['compound']
            count+=1
    return(sentiment_score/count)

if __name__ == '__main__':
    print(f"Sentiment score for cleaned text using vader: {vader_cleaned_score('input.txt')}")
    print(f"Sentiment score for raw text using vader:{vader_uncleaned_score('input.txt')}")



