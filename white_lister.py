# imports
import numpy as np
import pandas
import yfinance as yf
import datetime as dt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from pre_processors import *

def make_white_lister(ticker, deadgap, d_size=5, ddt_size=20, input_size=50, layer_sizes=[300, 200, 100, 50, 25, 10, 5]):
    """
    build a model to predict whether the price of a given stock will rise or fall over the course of a day

    inputs:
        ticker: string, the ticker of the company of interest
        input_size: integer, the number of days considered in the classification window

    returns:
        white_lister: model designed to predict the daily change in a stock's price
    """

    start = dt.datetime(1973, 1, 1, 0, 0, 0)
    data = yf.download(ticker, start=start)
    opens = data["Open"].to_numpy()
    closes = data["Close"].to_numpy()

    window = np.array(opens[:input_size])
    window /= np.max(window)
    if closes[input_size-1] > opens[input_size-1]:
        labels = np.array(0)
    elif closes[input_size-1] < opens[input_size-1]:
        labels = np.array(1)
    else:
        labels = np.array(2)

    for a in range(1,(len(opens) - (input_size-1))):
        window = np.vstack((window, opens[a:a+input_size]))
        window[a] /= np.max(window[a])
        #print(windows[a])
        if (closes[a+(input_size-1)] - opens[a+(input_size-1)]) > (deadgap*opens[a+(input_size-1)]):
            labels = np.append(labels, 0)
        elif (closes[a+(input_size-1)] - opens[a+(input_size-1)]) < (-deadgap*opens[a+(input_size-1)]):
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, 2)


    windows = []
    for a in range(len(labels)):
        windows.append(pre_process(window[a], d_size, input_size, ddt_size))

    windows = np.array(windows)
    
    train_windows, test_windows, train_labels, test_labels = train_test_split(windows, labels, train_size=0.7)

    hidden_layers = []
    for ls in layer_sizes:
        hidden_layers.append(tf.keras.layers.Dense(ls, activation ='sigmoid'))

    white_lister = tf.keras.models.Sequential([
        tf.keras.layers.Dense(len(windows[0]), activation='relu')] +
        hidden_layers + 
        [tf.keras.layers.Dense(3)])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    white_lister.compile(optimizer=opt,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

    print("\n\n")
    history = white_lister.fit(train_windows, train_labels, epochs=50, verbose=0)

    test_loss, test_acc = white_lister.evaluate(test_windows, test_labels, verbose=2)

    return white_lister, history, test_acc



if __name__ == "__main__":
    white_lister, history, test_acc = make_white_lister("GOOG", 0.015)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model training accuracy and loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()
    print(test_acc)