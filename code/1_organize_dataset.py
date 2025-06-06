import pandas as pd
import os

def get_dateset(df_stocks_day, df_gtrends, df_selic_cdi, target, ret_target, diff_gt):
    # Creating the target dataframe
    target_variable = df_stocks_day[target]
    if ret_target:
        target_variable = 100*target_variable.pct_change(1, fill_method=None) #daily return
    df_target = df_gtrends.copy(deep=True)
    if diff_gt:
        df_target = df_target.diff(1)
    df_target.columns  = ['gt ' + col for col in df_target.columns]
    gtrends            = df_target.columns
    df_target[target] = target_variable
    df_target.dropna(inplace=True)
    #Adding both SELC and CDI values as columns
    df_target["SELIC %d"] = df_selic_cdi[df_selic_cdi.columns[0]].shift(1)
    df_target["CDI %d"]   = df_selic_cdi[df_selic_cdi.columns[1]].shift(1)
    # Adding lagged values of the target as columns
    df_target["%s(-1)"%(target)] = df_target[target].shift(1)
    # Adding lagged values of the gtrends as columns
    for gtrend in gtrends:
        df_target[gtrend + "(-1)"] = df_target[gtrend].shift(1)
    # Dropping inserted NaNs
    df_target.dropna(inplace = True)
    # Dropping gtrends without lag columns
    df_target.drop(gtrends, axis = 1, inplace = True)
    return df_target

folder_path = 'Probest/TrabalhoFinal/Dados/'

df_gtrends    = pd.read_csv(folder_path+"gtrends_day_finance.csv", index_col="date")
df_selic_cdi  = pd.read_csv(folder_path+"SELIC_CDI.csv", sep=";", index_col="Date")[0:-1]
df_stocks_day = pd.read_csv(folder_path+"stocks_daily_close_prices.csv", index_col="Unnamed: 0")

df_selic_cdi.index  = pd.to_datetime(df_selic_cdi.index, dayfirst=True)
df_gtrends.index    = pd.to_datetime(df_gtrends.index)
df_stocks_day.index = pd.to_datetime(df_stocks_day.index)

target = "^BVSP"

df_target = get_dateset(df_stocks_day, df_gtrends, df_selic_cdi, target, False, False)
df_target.to_csv("Probest\TrabalhoFinal\Dados\price_gt\%s_dataset.csv"%target)
    
df_target = get_dateset(df_stocks_day, df_gtrends, df_selic_cdi, target, True, False)
df_target.to_csv("Probest\TrabalhoFinal\Dados\\ret_gt\%s_dataset.csv"%target)

df_target = get_dateset(df_stocks_day, df_gtrends, df_selic_cdi, target, False, True)
df_target.to_csv("Probest\TrabalhoFinal\Dados\price_delta\%s_dataset.csv"%target)

df_target = get_dateset(df_stocks_day, df_gtrends, df_selic_cdi, target, True, True)
df_target.to_csv("Probest\TrabalhoFinal\Dados\\ret_delta\%s_dataset.csv"%target)

