# imports
import numpy as np
import pandas
import yfinance as yf
import datetime as dt
#import tensorflow as tf

def make_white_lister(ticker, input_size):
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

    windows = np.array(opens[:input_size])
    print("\n hey \n")
    for a in range(1,(len(opens) - (input_size-1))):
        windows = np.vstack((windows, opens[a:a+input_size]))


    print(windows)




if __name__ == "__main__":
    make_white_lister("TSLA", 5)