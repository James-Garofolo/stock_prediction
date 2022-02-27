#imports
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import yfinance as yf
import datetime as dt
import tensorflow as tf

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

trend /= np.max(trend)

window = np.array(trend[0: 60])
if trend[59] > trend[60]:
    label = np.array(0)
else:
    label = np.array(1)
#need to make 60 minute arrays with 1 minute intervals between
for i in range(1,len(trend)- 60):
    window = np.vstack((window, trend[i:i + 60]))
    if trend[i+60]>trend[i+59]:
        label = np.append(label, 0)
    else:
        label = np.append(label, 1)
    
   
#print("\n window\n", window, "\n label\n", label)
#sets training percentage
train_window, test_window, train_label, test_label = train_test_split(window, label, train_size=0.7)

#neural network structure
daytrader = tf.keras.models.Sequential([
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dense( 100, activation = 'relu'),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(2)
])
 
 #now we need to add weights to this
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 
daytrader.compile(optimizer = opt,
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics =['accuracy'])

history = daytrader.fit(train_window, train_label, epochs=20 , verbose=1)

test_loss, test_acc = daytrader.evaluate(test_window, test_label, verbose=2)
print(test_acc)


  

