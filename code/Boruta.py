import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy


folder_path  = "Probest/TrabalhoFinal/Dados/"
results_path = "Probest/TrabalhoFinal/Resultados/"

targets = ["^BVSP", "BBAS3.SA", "BBDC4.SA", "BRL=X", "ITUB4.SA", "PETR4.SA", "SANB11.SA"]

def X_y_split(df, target):
    y = df[target].copy(deep=True)
    X = df.copy(deep=True)
    X = X.drop([target], axis = 1)
    return y, X

def map_rank_to_label(rank_value):
    if rank_value == 1:
        return "Selected"
    elif rank_value == 2:
        return "Tentative"
    else:
        return "Rejected"

for target in targets:
    df    = pd.read_csv(folder_path + target +"_dataset.csv")
    dates = df.pop("date")
    y, X    = X_y_split(df, target)
    X_names = X.columns
    rf            = RandomForestRegressor()
    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1, verbose = 2)
    feat_selector.fit(X.values, y.values)
    df_rank = pd.DataFrame({"X":X_names, "Rank":feat_selector.ranking_})
    df_rank['Status'] = df_rank['Rank'].apply(map_rank_to_label)
    df_rank.to_csv(results_path + "prices/boruta_%s.csv"%target)


for target in targets:
    df    = pd.read_csv(folder_path + target +"_dataset_return.csv")
    dates = df.pop("date")
    y, X    = X_y_split(df, target)
    X_names = X.columns
    rf            = RandomForestRegressor()
    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1, verbose = 2)
    feat_selector.fit(X.values, y.values)
    df_rank = pd.DataFrame({"X":X_names, "Rank":feat_selector.ranking_})
    df_rank['Status'] = df_rank['Rank'].apply(map_rank_to_label)
    df_rank.to_csv(results_path + "returns/boruta_%s.csv"%target)
