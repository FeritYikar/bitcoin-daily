# **Bitcoin Daily**

**Author:** _Ferit Yikar

<img src="images\btc.jpeg" width=50%>

## Overview
This project consists of 3 parts. The first part is a Time Series model that predictsnext days BTC/USD price. For data I use BTC/USD prices as well as Gold price, Nasdaq price, NYSE price and 10 year Treasury Yield. The second part is a Natural Language Processing model that understands the sentiment of a #Bitcoin or #BTC tweet. The last part is a Web App that retracts momentary market data and tweets, uses previous models and this data to give Bitcoin prediction and insight.
***
# Time Series

<img src="images\mape.png" width=100%>

For Time Series I used dataset I retrieved from Yahoo Finance. The Bitcoin dataset consists of bitcoin prices from September 17 2014 to March 1st 2022. For Exagenous data I have various market indexes. The model is built using Long Short Term Memory layered Recurrent Neural Network method.

***
## Data

The dataset consists of financial data retrieved from Yahoo Finance. 5 different datasets were used for the time series model, Bitcoin, NYSE, Nasdaq, Gold, 10 Year Treasury Yield from 09/17/2014 to 03/01/2022.

***
## Methods
This project uses Tensorflow Recurrent Neural Network method with Long Short Term Memory layers. When predicting Bitcoin prices for the next day, the data fram 60 previous days are used. For example first 60 days are used to predicts 61st day, data from day 2 to day 61 are used to predict price in 62nd day and so on. After trying different layers with and without dropout I found that one layer with 60 units(number of inputs, if exogenous data is used, add 60 per data) gives the best results. After running 8 models I found that the best results were achieved when I used Bitcoin prices with Nasdaq prices as exogenous data. However the val-loss scores were very close on many models. Using the time_series.ipynb file you can run your own models and compare results. For this file you can use the btc-ts environment inside environments folder.  
***
## Results

### Comparing Best Model Test Results with Actual Prices

My first step was to take a look at the model is doing on the test data

<img src="images\pred-actual.png" width=80%>

However this can be misleading since a little difference between prediction and actual seems fine on the graph, it can cause very big gains or losses. That is why I have ran 5 different back testings to see how would this model do if we had been using it vs other investment tools.

<img src="images\30days.png" width=40%>
<img src="images\60days.png" width=40%>
<img src="images\180days.png" width=40%>
<img src="images\365days.png" width=40%>
<img src="images\408days.png" width=40%>


Just like any financial tool, this model cannot guarantee best results all the time. When we look at different time periods within our test data we can say it is usually doing a good job.

# Natural Language Processing

<img src="images\twitter-btc.jpeg" width=100%>

***
## Data

For NLP part I used a dataset from data.world. The dataset consists of more than 50.000 tweets with hashtags #Bitcoin and/or #BTC. The dataset has 3 sentiments positive, negative and neutral. I will try to build a model to predict the sentiment of a tweet.

***
## Methods
This project uses Tensorflow Recurrent Neural Network method with Long Short Term Memory layers. Before any modelling can be done however, I have used different methods to turn tweets into arrays. For more details you can see NLP_tweets file and run your own model using btc-nlp environment. Once tweets are turned into arrays, the rnn model is applied. The best model with val-loss: 0.16812 is used for future predictions. Below we can see how the model does on the test data
***
## Results

### Confusion Matrix

The accuracy rate of the model on test data is 97%

<img src="images\confusion_matrix.png" width=80%>

When we look at the confusion matrix, we can see the model is doing a very good job predictiong all the sentiments. 


# Web App



## Data

<img src="images\web_app.png" width=50%>

For the web app to work sustainably, it should always retract data from APIs. The file retracts data online every 15 minutes.

### Updating the Data

<img src="images\terminal.png" width=80%>

Using the btc-env environment, update_file folder uses yfinance and tweepy libraries. The file first determines what is the latest financial data it has, then what is the date of today. After that it retracts the financial data from yfinance API. However on weekends although yfinance provides momentary prices for Bitcoin, it does not give Friday, Saturday and Sunday closing values until monday morning. Because of this this file also uses cryptocmd to get bitcoin prices when yfinance is not available. Once the data is retracted and added to the existing data, the model uses last 60 bitcoin prices as well as Nasdaq prices to predict next bitcoin price using previously saved best model. If after your modeling a different model becomes the best model, you should change the inputs and the model in the update_file file accordingly.  

After making the prediction the file retracts 500 most recent #Bitcoin tweets using tweepy library. Using the NLP model previously saved the file predicts the sentiment of all these tweets. Then we save the counts of 3 sentiments and saves them in a dictionary for web app to use. It also saves most retweeted tweets to display in the web app.

### Displaying the Data

Once we have all the data we have we can run our web_app file. This file uses streamlit library and should be ran using web streamlit environment.

## Next Steps

First next step is going live with the web app in the upcoming weeks.

- <u> Extend the predictions from Bitcoin </u>
    - NYSE
    - Nasdaq
    - More financial tools.
<br><br>

- <u>More frequent updates</u>
    - Twitter does not allow more than 1500 tweets per 15 minutes. With different account settings I can improve the model to run more frequently.
<br><br>
- <u>In the more advanced model of this web app you will be able to input your account balance and follow profits and possible future balances based on predictions</u>

***
## For More Information
Please review my full analysis in my files or my presentation.

For any additional questions, please contact

yikarferit@gmail.com <br />
https://www.linkedin.com/in/ferityikar/<br />

## Repository Structure

```
├── charts
├── data
├── dataframes
├── environments
├── models
├── .gitignore
├── NLP_tweets.ipynb
├── README.md                           
├── time_series.ipynb   
├── update_file.py                          
└── webapp.py
```
