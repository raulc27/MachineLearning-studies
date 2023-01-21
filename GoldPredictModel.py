from sklearn.linear_model import LinearRegression;
import pandas as pd;
import numpy as np;
import yfinance as yf


Df = yf.download('GLD','2008-01-01','2020-1-20', auto_adjust=True)
Df = Df[['Close']]
Df = Df.dropna()

# Define explanatory variables
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
y = Df['next_day_price']


# Split the data into train and test dataset

t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]


# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)

# Predicting the Gold ETF prices

predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price']
)

