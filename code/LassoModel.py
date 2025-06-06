import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

folder_path  = "Probest/TrabalhoFinal/Dados/"
results_path = "Probest/TrabalhoFinal/Resultados/"

targets = ["^BVSP", "BBAS3.SA", "BBDC4.SA", "BRL=X", "ITUB4.SA", "PETR4.SA", "SANB11.SA"]

def X_y_split(df, target):
    y = df[target].copy(deep=True)
    X = df.copy(deep=True)
    X = X.drop([target], axis = 1)
    return y, X

def scale_data(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = pd.DataFrame(scaler.transform(X))
    X_norm.columns  = [col for col in X.columns]
    return X_norm

for target in targets:
    df    = pd.read_csv(folder_path + target +"_dataset.csv")
    dates = df.pop("date")
    y, X = X_y_split(df, target)
    X_names = X.columns
    X_norm = scale_data(X)
    model = Lasso()
    model.fit(X_norm, y)
    df_selected_X_norm = pd.DataFrame({"X": X_names, "Selected":model.coef_ > 0})
    df_selected_X_norm.to_csv(results_path + "prices/lasso_%s.csv"%target)

for target in targets:
    df    = pd.read_csv(folder_path + target +"_dataset_return.csv")
    dates = df.pop("date")
    y, X = X_y_split(df, target)
    X_names = X.columns
    X_norm = scale_data(X)
    model = Lasso()
    model.fit(X_norm, y)
    df_selected_X_norm = pd.DataFrame({"X": X_names, "Selected":model.coef_ > 0})
    df_selected_X_norm.to_csv(results_path + "returns/lasso_%s.csv"%target)