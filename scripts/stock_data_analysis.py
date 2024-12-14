import os
import re
import numpy
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as pl
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load dataset
data = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\raw_analyst_ratings.csv")
print(data.head())
print(data.info())
# Check for missing values
print(data.isnull().sum())

# Check unique values in each column
print(data.nunique())
# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='ISO8601', errors='coerce')

# Check the range of dates
print(data['date'].min(), data['date'].max())

# Group by date to see publication frequency
date_counts = data['date'].value_counts().sort_index()
# Plot the distribution
plt.figure(figsize=(10, 5))
date_counts.plot(kind='line')
plt.title("Publication Frequency Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.show()
data['headline_word_count'] = data['headline'].apply(lambda x: len(str(x).split()))

# Plot word count distribution
data['headline_word_count'].plot(kind='hist', bins=20, figsize=(10, 5), color='purple')
plt.title("Headline Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()
# Join all headlines
all_headlines = " ".join(data['headline'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_headlines)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Check stock distribution
print(data['stock'].value_counts())

# Plot stock distribution
data['stock'].value_counts().plot(kind='bar', figsize=(10, 5), color='orange')
plt.title("Article Counts by Stock")
plt.xlabel("Stock")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.show()
# Group by stock and publisher
stock_publisher = data.groupby(['publisher', 'stock']).size().unstack(fill_value=0)

# Display heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(stock_publisher, cmap="Blues", annot=False)
plt.title("Article Distribution by Stock and Publisher")
plt.ylabel("Publisher")
plt.xlabel("Stock")
plt.show()
# Group by date and stock
stock_date = data.groupby(['date', 'stock']).size().unstack(fill_value=0)

# Plot time series for top stocks
stock_date.plot(figsize=(15, 7))
plt.title("Stock Mentions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Mentions")
plt.legend(title="Stocks", loc='upper left')
plt.show()
# Add sentiment scores
data['sentiment'] = data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Check sentiment distribution
data['sentiment'].plot(kind='hist', bins=20, figsize=(10, 5), color='green')
plt.title("Sentiment Distribution of Headlines")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Frequency")
plt.show()
#Top Modelling
# Convert headlines to a document-term matrix
vectorizer = CountVectorizer(stop_words='english')
headline_matrix = vectorizer.fit_transform(data['headline'])

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(headline_matrix)

# Display topics
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    print(f"Topic {i}:")
    print([words[i] for i in topic.argsort()[-10:]])
#Export or save analyzed data and result for future use
# Save dataset with new features
data.to_csv("analyzed_data.csv", index=False)
