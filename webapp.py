from tkinter import font
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from streamlit_autorefresh import st_autorefresh
import time
import datetime
from datetime import date
import warnings
import yfinance as yf
import tweepy
from cryptocmd import CmcScraper
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences


warnings.filterwarnings(action='ignore')


#Runs automatically after the wait period at the end
st_autorefresh(interval= 15 * 60 * 1000, key="dataframerefresh")


mpl.rcParams['font.size'] = 10

print('Welcome to Bitcoin Daily')

print('Reading the most recent dataset we have')
df_cumulative = pd.read_csv("dataframes/df_cumulative.csv", index_col='Date')

# Get the new data that was generated after the model last ran

last_day = df_cumulative.index[-1]

current_time = time.ctime()
current_hour = current_time[-13:-11]


    # yfinance uses london time, I am working in EST, therefore I will add 1 day if it is after 6pm, you can adjust this part accordingly

if int(current_hour) >= 18:
        today = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
else:
        today = date.today().strftime("%Y-%m-%d")

print(f'Today is {today}, retracting data from the last day we have data for: {last_day} until {today}.')

yahoo_data_raw = yf.download("^NDX ^NYA ^TNX ^XAU BTC-USD", start=last_day, end=today)

btc_current = yahoo_data_raw[(    'Close', 'BTC-USD')][-1]


# First day we already have, last day is the day we will predict, so we remove these 2. (when we use yesterday API gives missing data, thats why we use today and remove the last )
yahoo_data = yahoo_data_raw.iloc[1:-1, ]

# We use loc to choose only important columns and to order them.
yahoo_data = yahoo_data.loc[:,[(    'Close', 'BTC-USD'),(    'Close', '^NYA'),(    'Close', '^TNX'),(    'Close', '^XAU'),(    'Close', '^NDX')]]

# Add new rows to our existing df_cumulative dataframe
yahoo_data.index = yahoo_data.index.astype(str)
yahoo_data.columns = df_cumulative.columns

df_cumulative = pd.concat([df_cumulative, yahoo_data],axis=0)



print('Getting BTC prices from cryptocmd because yfinance does not work on the weekends')

scraper = CmcScraper("BTC")
crypto_df = scraper.get_dataframe()
crypto_df.Date = crypto_df.Date.map(lambda x: str(x)[:10])
crypto_df.set_index('Date', inplace=True)
most_recent_crypto = crypto_df.index[0]

try:
        missing_days = int(str(pd.to_datetime(most_recent_crypto) - pd.to_datetime(df_cumulative.index[-1]))[0])
except ValueError:
        missing_days = 0

if missing_days == 0:
        pass
else:
        print(f'We are still missing {missing_days} day(s) of data')

for i in range(missing_days):
        next_day = str(pd.to_datetime(df_cumulative.index[-1])+ datetime.timedelta(days=1))[:10]
        df_cumulative.loc[next_day] = [crypto_df['Close'][next_day],None,None,None,None]
print(f'Missing {missing_days} days of data is retrieved from another source')


print(f'Saving the Dataframe that is updated until {today}for future uses')

df_cumulative.fillna(method='ffill', inplace=True)
# Save the updated dataframe
df_cumulative.to_csv('dataframes/df_cumulative.csv')

print(f'Graphing 60 days BTC/USD trend until {today}')

# Graph last 60 days

last_2_months = df_cumulative.iloc[-60:]
last_2_months.BTC.plot(xticks=range(0,71,15), yticks=range(0,65000,10000))
plt.rcParams["figure.figsize"] = (26,10)
plt.ylabel('BTC/USD')
plt.savefig('charts/last_2_months.png')




print('Preparing the new data for predicting. Train-Test Split and Scaling')

size = 2314 #from time_series file

# Once the DF is updated we apply our data-prep to the new DF

train = df_cumulative.iloc[:size,]
test = df_cumulative.iloc[size:,]
train_scaled = train.copy() # to be chaged with scaled values
test_scaled = test.copy() # to be chaged with scaled values

# Scale the Data

btc_scaler = MinMaxScaler(feature_range=(0,1))
btc_train = btc_scaler.fit_transform(train.BTC.values.reshape(-1,1))
train_scaled.BTC = btc_train
btc_test = btc_scaler.transform(test.BTC.values.reshape(-1,1))
test_scaled.BTC = btc_test

nyse_scaler = MinMaxScaler(feature_range=(0,1))
nyse_train = nyse_scaler.fit_transform(train.NYSE.values.reshape(-1,1))
train_scaled.NYSE = nyse_train
nyse_test = nyse_scaler.transform(test.NYSE.values.reshape(-1,1))
test_scaled.NYSE = nyse_test

gold_scaler = MinMaxScaler(feature_range=(0,1))
gold_train = gold_scaler.fit_transform(train.Gold.values.reshape(-1,1))
train_scaled.Gold = gold_train
gold_test = gold_scaler.transform(test.Gold.values.reshape(-1,1))
test_scaled.Gold = gold_test

nasdaq_scaler = MinMaxScaler(feature_range=(0,1))
nasdaq_train = nasdaq_scaler.fit_transform(train.Nasdaq.values.reshape(-1,1))
train_scaled.Nasdaq = nasdaq_train
nasdaq_test = nasdaq_scaler.transform(test.Nasdaq.values.reshape(-1,1))
test_scaled.Nasdaq = nasdaq_test

int_rate_scaler = StandardScaler()
int_rate_train = int_rate_scaler.fit_transform(train.Int_Rate.values.reshape(-1,1))
train_scaled.Int_Rate = int_rate_train
int_rate_test = int_rate_scaler.transform(test.Int_Rate.values.reshape(-1,1))
test_scaled.Int_Rate = int_rate_test


print('Getting last 60 rows for predicting tomorrows price')

# Get the last 60 rows to use for predicting
input_df = test_scaled[['BTC','Nasdaq']].iloc[-60:,]

# Save the input as csv
input_df.to_csv('dataframes/input.csv')

print('Loading the best model that will be used for prediction')

#Load the saved best model
load_file = 'models/rnn_nasdaq.hdf5'
rnn_best_model = load_model(load_file)

print('Predicting tomorrows closing value')

# Make prediction on the best model
today_prediction_raw = rnn_best_model.predict(input_df.values.reshape(1,60,2))
today_prediction = btc_scaler.inverse_transform(today_prediction_raw)



# Twitter
consumer_key = 'input_personal_tweeter_developer_keys'
consumer_secret = 'input_personal_tweeter_developer_keys'
access_token = 'input_personal_tweeter_developer_keys'
access_token_secret = 'input_personal_tweeter_developer_keys'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

print('Getting 500 most recent tweets with hashtag #Bitcoin')

# Get 500 most recent #Bitcoin tweets
tweets = []
for tweet in tweepy.Cursor(api.search_tweets, q='#Bitcoin', lang='en').items(50):
        tweets.append(tweet)

# Get info from tweets
def timeline_df(tweets):
        id_list = [tweet.id for tweet in tweets]
        df = pd.DataFrame(id_list, columns = ['id'])
        
        df['Tweet']                  = [tweet.text for tweet in tweets]
        df['created_at']            = [tweet.created_at for tweet in tweets]
        df['Retweet_Count']         = [tweet.retweet_count for tweet in tweets]
        df['favorite_count']        = [tweet.favorite_count for tweet in tweets]
        df['source']                = [tweet.source for tweet in tweets]

        return df

twitter_df = timeline_df(tweets)

twitter_df.set_index('Retweet_Count', inplace=True)
top_tweets = twitter_df.sort_values('Retweet_Count',ascending=False)[['Tweet']].drop_duplicates().head(5)


print('Saving the most recent tweets as a CSV file')
twitter_df.to_csv('dataframes/tweets.csv')

print('Saving the top Tweets based on number of retweets.')

# Save top tweets to use later
top_tweets.to_csv('dataframes/top_tweets.csv')

# Save recent tweets sentiments to a dictionary

print('Cleaning the tweet data into stemmed words')
# Get cleaning_pipeline and tokenizer_lstm pickle and apply to new tweets
def remove_pattern(text, pattern):
        r = re.findall(pattern, text)
        for i in r:
            text = re.sub(i, '', text)
        return text

def pattern_apply(input):
        return input.str.replace('[^a-zA-Z]', ' ')

def remove_short(input, length=3):
        return input.apply(lambda x: ' '.join([w for w in x.split() if len(w) > length]))

def tokenize(input):
        return input.apply(lambda x: x.split())

def stemmer(input):
        return input.apply(lambda x: [SnowballStemmer('english').stem(i) for i in x])

def join_tokenize(input):
        return input.apply(lambda x: ' '.join(x))

def hashtag_extract(input, flatten=True):
        hashtags = []
        for i in input:
            ht = re.findall(r"#(\w+)", i)
            if flatten:
                hashtags.append(ht)
            else:
                hashtags.append([ht])

        return sum(hashtags, [])

def cleaning_pipeline(nlp_data):
        nlp_data['TweetClean'] = np.vectorize(remove_pattern)(nlp_data['Tweet'],"@[\w]*")
        nlp_data['TweetClean'] = np.vectorize(remove_pattern)(nlp_data['TweetClean'],'https?://[A-Za-z0-9./]+')
        nlp_data['TweetClean'] = pattern_apply(nlp_data['TweetClean'])
        nlp_data['TweetClean'] = remove_short(nlp_data['TweetClean'])
        tokenized_tweet = tokenize(nlp_data['TweetClean'])
        tokenized_tweet = stemmer(tokenized_tweet)
        nlp_data['TweetClean'] = join_tokenize(tokenized_tweet)
        nlp_data['TweetClean'] = np.vectorize(remove_pattern)(nlp_data['TweetClean'],"#[\w]*")
        nlp_data['Name Length'] = nlp_data['TweetClean'].str.len()

        return nlp_data

print_in = open('charts/tokenizer_lstm.pickle','rb')
tokenizer_lstm = pickle.load(print_in)
print_in.close()

print('Turning the tweets into arrays that can be processed')

twitter_df = cleaning_pipeline(twitter_df)

max_num_words = 50000
max_sequence_len = 112
embedding_dim = 100

X_instant = tokenizer_lstm.texts_to_sequences(twitter_df['TweetClean'].values)
X_instant = pad_sequences(X_instant, maxlen=max_sequence_len)

#Load the saved best model
load_file = 'models/nlp_model.hdf5'
nlp_model = load_model(load_file)

#Make Predictions
y_pred_instant = nlp_model.predict(X_instant)
y_pred_instant = pd.DataFrame(y_pred_instant, columns=['Negative', 'Neutral', 'Positive'])
y_pred_instant['Prediction'] = y_pred_instant.idxmax(1)
predictions = y_pred_instant['Prediction'].value_counts()

#Save the predictions to use in web app
sentiment_dict = {'Positive':predictions.Positive, 'Neutral': predictions.Neutral, 'Negative': predictions.Negative}

st.write('## Bitcoin Daily')
btc_image = Image.open('charts/bitcoin.jpeg')
st.image(btc_image)

st.write(f'#### BTC-USD: ${str(btc_current)[:8]}')



prediction = today_prediction[0][0]
# Put the prediction on the web app
st.write(f'Our Bitcoin prediction for {today} is ${str(prediction)[:8]}')


df_cumulative = pd.read_csv("dataframes/df_cumulative.csv", index_col='Date')

st.write('## BTC Trend')
# Last 60 days chart
image = Image.open('charts/last_2_months.png')
st.image(image)


st.write('## Current Bitcoin view on Twitter')

mpl.rcParams['font.size'] = 30

# Pie graph from Sentiment analysis
explode = (0.1, 0.2, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_dict.values(), explode=explode, labels=sentiment_dict.keys(), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.write(fig1)

st.write('## #Bitcoin')

top_tweets = pd.read_csv("dataframes/top_tweets.csv", index_col='Retweet_Count')

st.write(top_tweets)


st.write('## 7 Day Market Closing Info')
st.write(df_cumulative.tail(7))



st.write('##### Project by Ferit Yikar')
st.write('##### https://www.yikarferit.com')
st.write('###### https://github.com/FeritYikar')
st.write('###### yikarferit@gmail.com')


#The code ends here
print('The model will run again in 15 minutes')
