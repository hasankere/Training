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
data = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\AAPL_historical_data.csv")
print(data.head())
print(data.info())
