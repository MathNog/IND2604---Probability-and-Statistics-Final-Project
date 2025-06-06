import pytrends as pt
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

kw_list = ['Ibovespa', 'Bolsa de Valores', 'Ações', 'Dividendos', 'Renda Fixa', "Inflação", "CDI", "Dolar", "Bitcoin", "Renda Variável"]

initial_year = 2010
final_year   = 2024
dates = []
for year in range(initial_year, final_year):
    print(year)
    dates.append(str(year) + "-01-01 " + str(year+1) + "-01-01")


df_data = pd.DataFrame()
for i in range(0, len(kw_list), 5):
    print("Busca (%d)"%i)
    df_data_kw = pd.DataFrame()
    for timeframe in dates:
        print("   Ano %s"%timeframe)
        pytrend = TrendReq(backoff_factor=1, retries=5)
        pytrend.build_payload(kw_list[i:i+5], timeframe=timeframe, geo='BR')
        df_timeframe = pytrend.interest_over_time()
        df_data_kw = pd.concat([df_data_kw, df_timeframe], axis = 0)
        sleep(10)
    sleep(60)
    df_data = pd.concat([df_data, df_data_kw], axis = 1)
    df_data.drop("isPartial", axis=1, inplace=True)
    df_data.to_csv("Probest\TrabalhoFinal\Dados\gtrends_week_finance.csv")

plt.plot(df_data)
plt.show()