#imports
import numpy as np
import pandas
import yfinance as yf
import datetime as dt
#import tensorflow as tf

#make a function that returns model and test model
now=dt.datetime.now()
start = now - dt.timedelta(days = 30)
end = start + dt.timedelta(days = 7)
data=[]
while end < now:
    window = yf.download('F',start = start, end = end, interval = "1m" )
    data.append(window)
    start = end
    end = start + dt.timedelta(days = 7)
trend = data[0]['Open'].to_numpy()
for d in data[1:]:
    d = d['Open'].to_numpy()
    trend = np.hstack((trend, d))
print("dataframe\n\n", data)

window = np.array(trend[0: 60])
#need to make 60 minute arrays with 1 minute intervals between
for i in range(1,len(trend)- 60):
    window = np.vstack((window, trend[i:i + 60]))
   
print("array\n\n", window)
    




  

