import pandas as pd                        
from pytrends.request import TrendReq
from pytrends import *
from time import sleep
from datetime import datetime, timedelta


start_date = datetime(2010, 1, 1)
end_date = datetime(2024, 1, 1)
dates = []

# Generate the dates with intervals of 3 months
current_date = start_date
while current_date < end_date:
    dates.append((current_date.strftime("%Y-%m-%d") + " " + (current_date + timedelta(days=3*30)).strftime("%Y-%m-%d")))
    current_date += timedelta(days=3*30)


kw_list = ['bolsa de valores', 'crise economica', 'investimentos', 'ibovespa', 'b3', 'ações', 
            'renda fixa', 'renda variável', "banco do brasil", "btg", "itau", "bradesco", "economia", "bitcoin"]
  

df_results = pd.DataFrame()
df_errors = pd.DataFrame()
for i in range(0,len(kw_list),3):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=kw_list[i:i+3], geo="BR")
    try:
        related_queries = pytrend.related_queries()
        df_results = pd.concat([df_results,related_queries[kw_list[i]]["top"]])
        df_results = pd.concat([df_results,related_queries[kw_list[i+1]]["top"]])
        df_results = pd.concat([df_results,related_queries[kw_list[i+2]]["top"]])
        df_results.to_csv("Probest/TrabalhoFinal/Dados/keywords.csv")
    except:
        print("Esperando...")
        df_errors = pd.concat([df_errors,pd.DataFrame(kw_list[i:i+3])])
        df_errors.to_csv("Probest/TrabalhoFinal/Dados/errors_keywords.csv")
        sleep(1*60)

idx_to_drop = []
for i in range(0,df_errors.shape[0]):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=[df_errors.iloc[i,:].values[0]])
    try:
        related_queries = pytrend.related_queries()
        df_results = pd.concat([df_results,related_queries[df_errors.iloc[i,:].values[0]]["top"]])
        df_results.to_csv("Probest/TrabalhoFinal/Dados/keywords.csv")
        idx_to_drop.append(i)
    except:
        print("Esperando...")
        sleep(1*60)

df_errors = df_errors.drop(idx_to_drop)
if df_errors.shape[0] == 0:
    print("Foi tudo!")
    df_errors.to_csv("Probest/TrabalhoFinal/Dados/errors_keywords.csv")

df_results = pd.read_csv("Probest/TrabalhoFinal/Dados/keywords.csv", index_col="Unnamed: 0")

df_results_unique = df_results.groupby(["query"]).sum()

mean_value = df_results_unique.value.mean()

df_results_filtered = df_results_unique[df_results_unique.value > mean_value]

df_results_filtered["keyword"] = df_results_filtered.index
df_results_filtered.reset_index(drop=True, inplace=True)

for kw in kw_list:
    df_results_filtered.loc[len(df_results_filtered)] = [100, kw]

df_results_filtered.sort_values(by="value", ascending=False, inplace=True)
df_results_filtered.reset_index(drop=True, inplace=True)

df_results_filtered.to_csv("Probest/TrabalhoFinal/Dados/final_keywords.csv")