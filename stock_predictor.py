from daytrader import *
from white_lister import *
import os

a = True

ticker = input("Enter the ticker for the stock to predict: ")
while a:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        a = False
    except:
        ticker = input("The entered ticker was not found, please try again: ")

day_deadgap = float(input("What percentage change in stock price over a day would you like to consider significant? "))/100.0
min_deadgap = float(input("What percentage change in stock price over a day would you like to consider significant? "))/100.0

specify = input("Would you like to customize the network architecture? (y/n) ") == "y"
if specify:
    size = 1
    i = 1
    sizes = []
    while size > 0:
        size = int(input("Specify hidden layer " + str(i) + "'s size, or enter -1 to finish"))
        if size > 0:
            sizes.append(size)
    
    
    day_trade, dhist, dacc = make_daytrader(ticker, min_deadgap, layer_sizes=sizes)
    white_list, whist, wacc = make_white_lister(ticker, day_deadgap, layer_sizes=sizes)
else:   
    day_trade, dhist, dacc = make_daytrader(ticker, min_deadgap)
    white_list, whist, wacc = make_white_lister(ticker, day_deadgap)

day_trade.save(ticker + "_day_trader.h5")
white_list.save(ticker + "_white_lister.h5")
print("Saved a daily predictor model to " + os.getcwd() + " that can predict daily price changes of up to " + 
    str(day_deadgap*100) + "% with an accuracy of " + str(wacc*100) + "%\n")
print("Saved a day trader model to " + os.getcwd() + " that can predict price changes by the minute of up to " + 
    str(min_deadgap*100) + "% with an accuracy of " + str(dacc*100) + "%\n")
print("Happy trading!\n")

