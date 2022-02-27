#imports
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import yfinance as yf
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from pre_processors import *

#make a function that returns model and test model
def make_daytrader(ticker, deadgap, d_size=5, ddt_size=20, input_size=60, layer_sizes=[300, 200, 100, 50, 25, 10, 5]):
    now=dt.datetime.now()
    start = now - dt.timedelta(days = 30)
    end = start + dt.timedelta(days = 7)
    data=[]
    while end < now:
        window = yf.download(ticker,start = start, end = end, interval = "1m" )
        data.append(window)
        start = end
        end = start + dt.timedelta(days = 7)
    trend = data[0]['Open'].to_numpy()
    for d in data[1:]:
        d = d['Open'].to_numpy()
        trend = np.hstack((trend, d))

    window = np.array(trend[0: input_size])
    window /= np.max(window)
    if (trend[input_size] - trend[input_size-1]) > (deadgap*trend[input_size-1]):
        label = np.array(0)
    elif (trend[input_size] - trend[input_size-1]) < (-deadgap*trend[input_size-1]):
        label = np.array(1)
    else:
        label = np.array(2)

    #need to make 60 minute arrays with 1 minute intervals between
    for i in range(1,len(trend)- input_size):
        window = np.vstack((window, trend[i:i + input_size]))
        window[i] /= np.max(window[i])

        diff = trend[i+input_size]-trend[i+input_size-1]
        if diff > (deadgap*trend[i+input_size-1]):
            label = np.append(label, 0)
        elif diff < (-deadgap*trend[i+input_size-1]):
            label = np.append(label, 1)
        else:
            label = np.append(label, 2)
        
    windows = []
    for a in range(len(label)):
        windows.append(pre_process(window[a], d_size, input_size, ddt_size))
    
    #print("\n window\n", window, "\n label\n", label)
    #sets training percentage
    train_window, test_window, train_label, test_label = train_test_split(window, label, train_size=0.7)

    hidden_layers = []
    for ls in layer_sizes:
        hidden_layers.append(tf.keras.layers.Dense(ls, activation ='sigmoid'))

    windows = np.array(windows)

    #neural network structure
    daytrader = tf.keras.models.Sequential([
        tf.keras.layers.Dense(len(windows[0]), activation='relu')] +
        hidden_layers + 
        [tf.keras.layers.Dense(3)])
    
    #now we need to add weights to this
    opt = tf.keras.optimizers.Adam(learning_rate=0.000025) 
    daytrader.compile(optimizer = opt,
                        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics =['accuracy'])

    history = daytrader.fit(train_window, train_label, epochs=20 , verbose=0)

    test_loss, test_acc = daytrader.evaluate(test_window, test_label, verbose=2)

    return daytrader, history, test_acc


if __name__ == "__main__":
    daytrader, history, test_acc = make_daytrader("F", 0.00125,)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model training accuracy and loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()
    print(test_acc)


  

