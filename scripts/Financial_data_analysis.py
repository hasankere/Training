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

# Define file paths for the datasets
%matplotlib inline
file_paths = [
    "C:\\Users\\Hasan\\Desktop\\data science folder\\AAPL_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\AMZN_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\GOOG_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\META_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\MSFT_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\NVDA_historical_data.csv",
    "C:\\Users\\Hasan\\Desktop\\data science folder\\TSLA_historical_data.csv"
]

stock_names = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]

# Load datasets and add a 'Stock' column
dataframes = []
for file, stock in zip(file_paths, stock_names):
    df = pd.read_csv(file)
    df['Stock'] = stock  # Add stock name to differentiate
    dataframes.append(df)

# Combine all datasets into a single DataFrame
combined_data = pd.concat(dataframes, ignore_index=True)

# Display basic information
print(combined_data.head())
# Combine all datasets
combined_data = pd.concat(dataframes, ignore_index=True)
combined_data.sort_values(by=['Stock', 'Date'], inplace=True)

# Function to calculate financial indicators
def calculate_indicators(df):
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)
    df['Daily_Return'] = df['Close'].pct_change()
    return df

# Apply indicators
analyzed_data = combined_data.groupby('Stock', group_keys=False).apply(calculate_indicators)

# Save the analyzed data
output_file = "C:\\Users\\Hasan\\Desktop\\data science folder\\analyzed_financial_data.csv"
analyzed_data.to_csv(output_file, index=False)
print(f"Analyzed data saved to {output_file}")
# Visualization
def plot_stock_with_indicators(stock):
    stock_data = analyzed_data[analyzed_data['Stock'] == stock]
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['Date'], stock_data['SMA_20'], label='20-Day SMA', color='orange')
    plt.plot(stock_data['Date'], stock_data['SMA_50'], label='50-Day SMA', color='green')
    plt.fill_between(stock_data['Date'], stock_data['BB_upper'], stock_data['BB_lower'], color='gray', alpha=0.3, label='Bollinger Bands')
    plt.title(f'{stock} Price, Moving Averages, and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    # RSI
    plt.figure(figsize=(14, 4))
    plt.plot(stock_data['Date'], stock_data['RSI'], color='purple', label='RSI')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'{stock} Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid()
    plt.show()
    # Plot for each stock
for stock in stock_names:
    plot_stock_with_indicators(stock)

# Fetch summary statistics using yfinance
print("\nStock Summary using yfinance:")
for stock in stock_names:
    ticker = yf.Ticker(stock)
    info = ticker.info
    print(f"\n{stock} Summary:")
    print(f"Current Price: {info.get('currentPrice', 'N/A')}")
    print(f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
    print(f"Market Cap: {info.get('marketCap', 'N/A')}")
    news_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\raw_analyst_ratings.csv"
# Load news dataset
news_data = pd.read_csv(news_path)

# Ensure 'Date' column exists and is a datetime object
news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')

# Drop rows where Date is invalid
news_data = news_data.dropna(subset=['date'])

print("News data preview:")
print(news_data.head())
# Load news data
news_data = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\raw_analyst_ratings.csv")

# Debugging: Print column names to identify the correct 'Date' column
print("News Data Columns:", news_data.columns)

# Check for column names like 'date', 'DATE', or others and normalize
if 'date' in news_data.columns:
    news_data.rename(columns={'date': 'Date'}, inplace=True)
elif 'DATE' in news_data.columns:
    news_data.rename(columns={'DATE': 'Date'}, inplace=True)
else:
    print("Error: No 'Date' column found in the news dataset.")
    raise KeyError("No column named 'Date' in the dataset.")

# Convert 'Date' column to datetime
news_data['Date'] = pd.to_datetime(news_data['Date'], errors='coerce')

# Drop invalid dates
news_data.dropna(subset=['Date'], inplace=True)

# Preview the cleaned news data
print(news_data.head())
# Check the date range for stock data
print("Stock Data Date Range:")
print(combined_data['Date'].min(), combined_data['Date'].max())

# Check the date range for news data
print("News Data Date Range:")
print(news_data['Date'].min(), news_data['Date'].max())

# Inspect a sample of the dates
print("Sample Dates from Stock Data:")
print(combined_data['Date'].dropna().sort_values().unique()[:10])

print("Sample Dates from News Data:")
print(news_data['Date'].dropna().sort_values().unique()[:10])
# Filter stock data to the news data date range
stock_data_filtered = combined_data[
    (combined_data['Date'] >= news_data['Date'].min()) & 
    (combined_data['Date'] <= news_data['Date'].max())
]

print("Filtered Stock Data Date Range:")
print(stock_data_filtered['Date'].min(), stock_data_filtered['Date'].max())

# Check the filtered stock data
print(stock_data_filtered.head())
#for Corelation merge the news and stock data using date
# Standardize both Date columns to remove timestamps (keep only the date)
stock_data_filtered['Date'] = stock_data_filtered['Date'].dt.date
news_data['Date'] = news_data['Date'].dt.date

# Convert both 'Date' columns back to datetime objects for consistency
stock_data_filtered['Date'] = pd.to_datetime(stock_data_filtered['Date'])
news_data['Date'] = pd.to_datetime(news_data['Date'])

# Merge datasets again on 'Date'
aligned_data = pd.merge(stock_data_filtered, news_data, on='Date', how='inner')

# Check the result
print("Aligned Data Sample:")
print(aligned_data.head())

print("Aligned Data Shape:", aligned_data.shape)
# Define a function to get sentiment
def get_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0  # Neutral if no text
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to headlines
aligned_data['sentiment'] = aligned_data['headline'].apply(get_sentiment)

# Classify sentiment into positive, neutral, and negative
def classify_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

aligned_data['sentiment_label'] = aligned_data['sentiment'].apply(classify_sentiment)

# Check the results
print("Sentiment Analysis Results:")
print(aligned_data[['headline', 'sentiment', 'sentiment_label']].head())
# Plot sentiment distribution
sentiment_counts = aligned_data['sentiment_label'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['Purple', 'grey', 'red'])
plt.title('Sentiment Distribution in News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()
# Assuming 'Close' is the column containing daily closing prices in aligned_data

# Calculate Daily Stock Returns as percentage change
aligned_data['Daily Return'] = aligned_data['Close'].pct_change() * 100

# Display the updated DataFrame with Daily Returns
print("Aligned Data with Daily Stock Returns:")
print(aligned_data.head())

# Optional: Check for missing values or irregularities
print("Daily Returns Calculated Successfully.")
# Merge the datasets
aligned_data = pd.merge(stock_data_filtered, news_data, on='Date', how='inner')

# Display the columns
print("Columns in merged dataset:", aligned_data.columns)
# Define a function to calculate sentiment polarity
def calculate_sentiment(text):
    if pd.isna(text):
        return 0  # Neutral sentiment for missing headlines
    return TextBlob(text).sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

# Apply the function to the 'headline' column
aligned_data['Sentiment'] = aligned_data['headline'].apply(calculate_sentiment)

# Check the updated dataset
print("Dataset with Sentiment Scores:")
print(aligned_data[['Date', 'headline', 'Sentiment']].head())
# Ensure 'aligned_data' has Date and Sentiment columns
print("Columns in merged dataset:", aligned_data.columns)

# Group by Date and compute the average sentiment
average_sentiment = aligned_data.groupby('Date', as_index=False)['Sentiment'].mean()

# Rename the column for clarity
average_sentiment.rename(columns={'Sentiment': 'Average Sentiment'}, inplace=True)

# Merge the average sentiment back into the original aligned_data (optional step)
aligned_data = pd.merge(aligned_data, average_sentiment, on='Date', how='left')

# Drop duplicates if necessary
aligned_data = aligned_data.drop_duplicates()

# Display the updated dataset
print("Updated Merged Dataset with Average Sentiment:")
print(aligned_data.head())
from scipy.stats import pearsonr

# Check for missing values and drop rows with NaNs in relevant columns
aligned_data = aligned_data.dropna(subset=['Adj Close', 'Sentiment'])

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(aligned_data['Adj Close'], aligned_data['Sentiment'])

# Display the correlation result
print("Correlation Analysis between Daily Returns and News Sentiment:")
print(f"Pearson Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("The correlation is statistically significant at the 5% level.")
else:
    print("The correlation is not statistically significant.")
    # Scatter plot of Sentiment vs Daily Returns
plt.figure(figsize=(8, 6))
plt.scatter(aligned_data['Sentiment'], aligned_data['Adj Close'], color='blue', alpha=0.5)
plt.title("Scatter Plot of Sentiment vs Daily Returns")
plt.xlabel("Sentiment Score")
plt.ylabel("Daily Stock Return (%)")
plt.grid()
plt.show()