import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
df = pd.read_csv('./house price/data/houseprice.csv')
df = df.fillna(0)
#By setting errors=’coerce’, you’ll transform the non-numeric values into NaN.
df['area'] = pd.to_numeric(df['area'],errors='coerce')
#Use dropna with parameter subset for specify column for check NaNs:
df = df.dropna(subset=['area'])
data = df.reset_index(drop=True)
data.to_csv('houseprice_new.csv')

y = data['price']
X = data.drop(columns='price', axis=1)
 
# Splitting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
 
# Making the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the output
y_pred = lr.predict(X_test)
 
# Saving the model
import joblib
 
joblib.dump(lr, "lr_model.sav")

