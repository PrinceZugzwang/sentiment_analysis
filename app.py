import streamlit as st
import requests
from textblob import TextBlob

# Define the base URL for the news API
NEWS_API_BASE_URL = 'https://newsapi.org/v2/everything'

# Define your news API key here
NEWS_API_KEY = '2821a8e0b887455aac328cd7f45b0483'

def get_stock_news(stock_symbol):
    """
    Fetches news articles related to the given stock symbol from the News API.
    """
    # Define the parameters for the news API request
    params = {
        'q': stock_symbol,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'relevancy'
    }

    # Send a GET request to the news API
    response = requests.get(NEWS_API_BASE_URL, params=params)

    # If the request was successful, return the articles
    if response.status_code == 200:
        articles = response.json()['articles']
        return articles

    # If the request was not successful, return an error message
    else:
        return [{'title': 'Error', 'description': 'Unable to fetch news articles'}]

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using TextBlob.
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Streamlit app
st.title("Stock Sentiment Analysis")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")

# Fetch news articles
news_articles = get_stock_news(stock_symbol)

# Display news articles
st.subheader("Latest News Articles:")
for article in news_articles:
    st.write(f"**Title:** {article['title']}")
    st.write(f"**Description:** {article['description']}")
    st.write(f"**URL:** {article['url']}")
    st.write("---")

# Analyze sentiment of news articles
sentiment_scores = [analyze_sentiment(article['title']) for article in news_articles]
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

# Display overall sentiment
st.subheader("Overall Sentiment Analysis:")
st.write(f"Average Sentiment Score: {average_sentiment}")

# Interpret sentiment
if average_sentiment > 0:
    st.write("Overall sentiment is positive!")
elif average_sentiment < 0:
    st.write("Overall sentiment is negative!")
else:
    st.write("Overall sentiment is neutral.")
