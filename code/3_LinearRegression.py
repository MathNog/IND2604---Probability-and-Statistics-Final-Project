import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def get_fitted_params_summary(fitted_model):
    df_summary = pd.DataFrame({
        "coef": fitted_model.params,
        "std err": fitted_model.bse,
        "t": fitted_model.tvalues,
        "p-value": fitted_model.pvalues,
        "LB": fitted_model.conf_int()[0],
        "UB": fitted_model.conf_int()[1]
    })
    return df_summary

def get_F_Test_full(fitted_model):
    df_f = pd.DataFrame({
        "F":[fitted_model.fvalue],
        "p-value": [fitted_model.f_pvalue]
    })
    return df_f


def residuals_HP(fitted_model):
    std_residuals = fitted_model.resid_pearson
    # Ljung Box
    lb_test = acorr_ljungbox(std_residuals, [30])
    # Jarque Bera
    jb_test = jarque_bera(std_residuals)
    # Breusch Pagan
    bp_test = het_breuschpagan(fitted_model.resid, fitted_model.model.exog)
    df_hp = pd.DataFrame({
        "JarqueBera": [jb_test[1]],
        "LjungBox": [lb_test.lb_pvalue.values[0]],
        "BreushPagan": [bp_test[1]]
    })
    return df_hp

def get_metrics(fitted_model, y):
    df_metrics = pd.DataFrame({
        "MSE":  [mean_squared_error(y, fitted_model.fittedvalues)],
        "RMSE": [np.sqrt(mean_squared_error(y, fitted_model.fittedvalues))],
        "MAPE": [mean_absolute_percentage_error(y, fitted_model.fittedvalues)],
        "MAE":  [mean_absolute_error(y, fitted_model.fittedvalues)],
        "R2": [fitted_model.rsquared],
        "AdjustedR2": [fitted_model.rsquared_adj],
        "AIC":[fitted_model.aic],
        "BIC":[fitted_model.bic]
    })
    return df_metrics

def get_outliers_indexes(fitted_model, N):
    outlier_idx = np.argsort(np.abs(std_residuals))[-N:]
    return outlier_idx


def create_dummy_variables(X, outlier_idx):
    T = X.shape[0]
    N = len(outlier_idx)
    D = pd.DataFrame(np.zeros((T,0)))
    for i in range(1, N+1):
        D[f'dummy_{i}'] = 0
    for (i, index) in enumerate(outlier_idx):
        D.at[index, f'dummy_{i+1}'] = 1
    return D

def concat_dummy(X, D):
    X_dummy = pd.concat([X, D], axis=1)
    return X_dummy


def F_Test_gt(fitted_model):
    hypotheses = ''
    for param in fitted_model.params.index:
        if "gt" in param:
            hypotheses += '(%s = 0), '%param
    hypotheses = hypotheses[0:-2]
    f_test = fitted_model.f_test(hypotheses)
    df_f_gt = pd.DataFrame({
        "F":[f_test.fvalue],
        "pvalue":[f_test.pvalue]
    })
    return df_f_gt

def plot_fac(fitted_model, outlier):
    std_residuals = fitted_model.resid_pearson
    p = plt.figure(figsize=(10,5))
    p = plot_acf(std_residuals)
    p = plt.ylim(-0.15, 0.15)
    p = plt.title(f"Autocorrelação dos resíduos padronizados {outlier}")
    p = plt.xlabel('Lags')
    p = plt.ylabel('FAC')
    return p

def plot_residuals(fitted_model, outlier):
    std_residuals = fitted_model.resid_pearson
    p = plt.figure(figsize=(10,5))
    p = plt.plot(std_residuals)
    p = plt.title(f"Resíduos padronizados {outlier}")
    return p

def plot_qqplot(fitted_model, outlier):
    p = plt.figure(figsize=(10,5))
    p = sm.qqplot(std_residuals, line="45")
    p = plt.xlim(-6, 6)
    p = plt.title(f"QQPlot dos resíduos padronizado {outlier}")
    return p

plt.style.use('seaborn-v0_8')

folder_path  = "Probest/TrabalhoFinal/Dados/"
results_path = "Probest/TrabalhoFinal/Resultados/"

target = "^BVSP"

scale = "price_gt"

############### Preparando datasets ####################
df = pd.read_csv(folder_path + "\%s\%s_dataset.csv"%(scale, target))
dates  = df.pop("date")
y      = df.loc[:,"^BVSP"]
X      = df.iloc[:, 1:]
X      = sm.add_constant(X)
print(X.shape)
############### Estimando modelos ####################
model        = sm.OLS(y, X)
fitted_model = model.fit()



############### Valores estimados ####################

df_summary = get_fitted_params_summary(fitted_model)
df_summary.to_csv(results_path + "\%s\linear_regression\%s_fitted_params.csv"%(scale, target))

# Teste F
df_f = get_F_Test_full(fitted_model)
df_f.to_csv(results_path + "\%s\linear_regression\%s_F_test.csv"%(scale, target))


############### Análise dos resíduos ####################

############### Testes de Hipótese dos resíduos ####################


df_hp = residuals_HP(fitted_model)
df_hp.to_csv(results_path+"\%s\linear_regression\hypothesis_test_std_residuals.csv"%scale)

# Métricas de Erro
df_metrics = get_metrics(fitted_model, y)
df_metrics.to_csv(results_path+"\%s\linear_regression\error_metrics.csv"%scale)



############### Gráficos dos resíduos ####################

# FAC
plot_fac(fitted_model)
plt.savefig(results_path+"\%s\linear_regression\\fac_std_residuals.jpg"%scale)
plt.close()

# Resíduos
plot_residuals(fitted_model, "")
plt.savefig(results_path+"\%s\linear_regression\std_residuals.jpg"%scale)
plt.close()

# QQPlot
plot_qqplot(fitted_model, "")
plt.savefig(results_path+"\%s\linear_regression\qqplot_std_residuals.jpg"%scale)
plt.close()
    


############### Teste F para apenas as variáveis do GT ####################


df_f_gt = F_Test_gt(fitted_model)
df_f_gt.to_csv(results_path+"\%s\linear_regression\\f_test_gt.csv"%scale)


" ############### Tratamento de outliers #################### "


outlier_idx = get_outliers_indexes(fitted_model, 7)
D = create_dummy_variables(X, outlier_idx)
X_outlier = concat_dummy(X, D)

model_outlier        = sm.OLS(y, X_outlier)
fitted_model_outlier = model_outlier.fit()


############### Valores estimados ####################

df_summary = get_fitted_params_summary(fitted_model_outlier)
df_summary.to_csv(results_path + "\%s\linear_regression\%s_fitted_params_outlier.csv"%(scale, target))

# Teste F
df_f = get_F_Test_full(fitted_model_outlier)
df_f.to_csv(results_path + "\%s\linear_regression\%s_F_test_outlier.csv"%(scale, target))


############### Análise dos resíduos ####################

############### Testes de Hipótese dos resíduos ####################


df_hp = residuals_HP(fitted_model_outlier)
df_hp.to_csv(results_path+"\%s\linear_regression\hypothesis_test_std_residuals_outlier.csv"%scale)

# Métricas de Erro
df_metrics = get_metrics(fitted_model_outlier, y)
df_metrics.to_csv(results_path+"\%s\linear_regression\error_metrics_outlier.csv"%scale)


############### Gráficos dos resíduos ####################

# FAC
plot_fac(fitted_model_outlier, " - pós tratamento de outliers")
plt.savefig(results_path+"\%s\linear_regression\\fac_std_residuals_outlier.jpg"%scale)
plt.close()

# Resíduos
plot_residuals(fitted_model_outlier, " - pós tratamento de outliers")
plt.savefig(results_path+"\%s\linear_regression\std_residuals_outlier.jpg"%scale)
plt.close()

# QQPlot
plot_qqplot(fitted_model_outlier, " - pós tratamento de outliers")
plt.savefig(results_path+"\%s\linear_regression\qqplot_std_residuals_outlier.jpg"%scale)
plt.close()
    
############### Teste F para apenas as variáveis do GT ####################


df_f_gt = F_Test_gt(fitted_model_outlier)
df_f_gt.to_csv(results_path+"\%s\linear_regression\\f_test_gt_outlier.csv"%scale)
