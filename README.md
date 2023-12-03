Link to demo: https://www.loom.com/share/8c3c728e06354cf4a4f305c9fa926484?sid=aca16896-eaa2-49ca-ade3-ad5182ca0b54 
# Stock Sentiment Analysis App

This is a Streamlit web application for analyzing stock sentiment using Natural Language Processing (NLP) techniques. The application fetches news articles related to a given stock symbol, performs sentiment analysis, and provides visualizations for user interpretation.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- Fetch news articles related to a specific stock symbol.
- Perform sentiment analysis on the news articles using TextBlob and VADER sentiment analysis.
- Display the overall sentiment and sentiment scores of individual articles.
- Visualize sentiment scores with bar charts.
- Identify entities (organizations and people) mentioned in the news articles using spaCy NER.
- Apply topic modeling techniques (Latent Dirichlet Allocation) to identify main topics in the news articles.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
