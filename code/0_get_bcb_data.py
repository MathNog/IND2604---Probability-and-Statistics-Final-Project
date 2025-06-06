import pandas as pd
from bcb import sgs
import matplotlib.pyplot as plt

# Busca a série do IPCA e IGP-M
ifl_month = sgs.get({'ipca': 433,
                    'igp-m': 189}, start = '2010-01-01')

data_day = sgs.get({'meta selic': 432,
                    "selic":11,
                    "cdi": 12}, start = '2010-01-01')

# Transforma a frequência da data em mensal
ifl_month.index = ifl_month.index.to_period('M')
data_day.index  = data_day.index.to_period('D')

ifl_month.isna().sum()

data_day.isna().sum()
data_day.dropna(inplace = True)

data_day.to_csv("Probest/TrabalhoFinal/Dados/dados_bsb_diarios.csv")