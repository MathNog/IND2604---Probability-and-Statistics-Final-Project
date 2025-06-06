import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt


startDate = datetime.datetime(2010, 1, 1)
endDate   = datetime.datetime(2024, 1, 10)
tickers = ["ITUB4.SA", "PETR4.SA", "SANB11.SA", "BBAS3.SA", "BBDC4.SA", "^BVSP", "BRL=X"]

df_close = pd.DataFrame()
for ticker in tickers[0:-1]:
    df_ticker = yf.download(ticker, start=startDate, end=endDate)
    df_close = pd.concat([df_close, df_ticker.loc[:,"Adj Close"]], axis=1)


df_dollar = yf.download(tickers[-1], start=startDate, end=endDate)["Adj Close"]
df_close = pd.concat([df_close, df_dollar], axis = 1, join = "inner")

df_close.columns = tickers
df_close.rename(columns={"BRL=X": "CotacaoDolar"})

df_close.to_csv("Probest/TrabalhoFinal/Dados/stocks_daily_close_prices.csv")



