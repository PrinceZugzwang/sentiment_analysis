import tweepy
import streamlit as st
import requests
import spacy
from textblob import TextBlob
from gensim import corpora, models
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import VADER sentiment analysis model
import nltk

# Define the base URL for the news API
NEWS_API_BASE_URL = 'https://newsapi.org/v2/everything'

# Define your news API key here
NEWS_API_KEY = '2821a8e0b887455aac328cd7f45b0483'

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Download the VADER lexicon data
nltk.download('vader_lexicon')

# Function to fetch news articles related to the given stock symbol from the News API
def get_stock_news(stock_symbol):
    params = {
        'q': stock_symbol,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'relevancy'
    }
    response = requests.get(NEWS_API_BASE_URL, params=params)

    if response.status_code == 200:
        articles = response.json()['articles']
        return articles
    else:
        return [{'title': 'Error', 'description': 'Unable to fetch news articles'}]

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)['compound']
    return sentiment_score

# Streamlit app
st.title("Stock Sentiment Analysis with NER, Topic Modeling, and Financial Data")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")

# Fetch news articles
news_articles = get_stock_news(stock_symbol)

# Display the top 5 news articles
st.subheader("Top 5 News Articles:")
for article in news_articles[:5]:
    st.write(f"**Title:** {article['title']}")
    st.write(f"**Description:** {article['description']}")

    # Extract entities using spaCy NER
    doc = nlp(article['title'])
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON']]
    st.write(f"**Entities:** {entities}")

    st.write(f"**URL:** {article['url']}")
    st.write("---")

# Analyze sentiment of the top 5 news articles using TextBlob
sentiment_scores_textblob = [analyze_sentiment_textblob(article['title']) for article in news_articles[:5]]
average_sentiment_textblob = sum(sentiment_scores_textblob) / len(sentiment_scores_textblob)

# Display overall sentiment using TextBlob
st.subheader("Overall Sentiment Analysis (TextBlob):")
st.write(f"Average Sentiment Score (TextBlob): {average_sentiment_textblob}")

# Interpret sentiment using TextBlob
if average_sentiment_textblob > 0:
    st.write("Overall sentiment (TextBlob) is positive!")
elif average_sentiment_textblob < 0:
    st.write("Overall sentiment (TextBlob) is negative!")
else:
    st.write("Overall sentiment (TextBlob) is neutral.")

# Analyze sentiment of the top 5 news articles using VADER
sentiment_scores_vader = [analyze_sentiment_vader(article['title']) for article in news_articles[:5]]
average_sentiment_vader = sum(sentiment_scores_vader) / len(sentiment_scores_vader)

# Display overall sentiment using VADER
st.subheader("Overall Sentiment Analysis (VADER):")
st.write(f"Average Sentiment Score (VADER): {average_sentiment_vader}")

# Interpret sentiment using VADER
if average_sentiment_vader > 0:
    st.write("Overall sentiment (VADER) is positive!")
elif average_sentiment_vader < 0:
    st.write("Overall sentiment (VADER) is negative!")
else:
    st.write("Overall sentiment (VADER) is neutral.")

# Create a DataFrame for visualization
df_sentiment_textblob = pd.DataFrame({'Article': [f"Article {i+1}" for i in range(5)], 'Sentiment (TextBlob)': sentiment_scores_textblob})
df_sentiment_vader = pd.DataFrame({'Article': [f"Article {i+1}" for i in range(5)], 'Sentiment (VADER)': sentiment_scores_vader})

# Plot bar charts
st.subheader("Sentiment Scores Visualization:")
fig_sentiment, (ax_textblob, ax_vader) = plt.subplots(1, 2, figsize=(12, 5))

# TextBlob
ax_textblob.bar(df_sentiment_textblob['Article'], df_sentiment_textblob['Sentiment (TextBlob)'],
                color=['green' if score > 0 else 'red' if score < 0 else 'gray' for score in sentiment_scores_textblob])
ax_textblob.set_ylabel('Sentiment Score')
ax_textblob.set_title('Sentiment Scores (TextBlob) of Top 5 Articles')

# VADER
ax_vader.bar(df_sentiment_vader['Article'], df_sentiment_vader['Sentiment (VADER)'],
             color=['green' if score > 0 else 'red' if score < 0 else 'gray' for score in sentiment_scores_vader])
ax_vader.set_ylabel('Sentiment Score')
ax_vader.set_title('Sentiment Scores (VADER) of Top 5 Articles')

st.pyplot(fig_sentiment)

# Preprocess the text for Topic Modeling
texts = [article['title'] for article in news_articles[:5]]

# Tokenize the text
tokenized_text = [text.split() for text in texts]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(tokenized_text)

# Create a corpus
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# Train the LDA model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Display the topics
st.subheader("Topics Discussed in the News Articles:")
for i, topic in enumerate(lda_model.print_topics()):
    st.write(f"Topic {i + 1}: {topic[1]}")
