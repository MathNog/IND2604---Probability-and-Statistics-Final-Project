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

plt.style.use('seaborn-v0_8')

folder_path  = "Probest/TrabalhoFinal/Dados/"
results_path = "Probest/TrabalhoFinal/Resultados/"

target = "^BVSP"

scales = ["price_gt", "price_delta", "ret_gt", "ret_delta"]

fitted_models = []

for scale in scales:
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
    fitted_models.append(fitted_model)


############### Valores estimados ####################

for (i, fitted_model) in enumerate(fitted_models):
    scale = scales[i]
    df_summary = pd.DataFrame({
        "coef": fitted_model.params,
        "std err": fitted_model.bse,
        "t": fitted_model.tvalues,
        "p-value": fitted_model.pvalues,
        "LB": fitted_model.conf_int()[0],
        "UB": fitted_model.conf_int()[1]
    })
    df_summary.to_csv(results_path + "\%s\linear_regression\%s_fitted_params.csv"%(scale, target))
    # Teste F
    df_f = pd.DataFrame({
        "F":[fitted_model.fvalue],
        "p-value": [fitted_model.f_pvalue]
    })
    df_f.to_csv(results_path + "\%s\linear_regression\%s_F_test.csv"%(scale, target))



############### Análise dos resíduos ####################

############### Testes de Hipótese dos resíduos ####################


# Juntar tudo num dataframe apenas
for (i, fitted_model) in enumerate(fitted_models):
    residuals = fitted_model.resid
    std_residuals = fitted_model.resid_pearson
    y = fitted_model.model.endog
    scale = scales[i]
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
    df_hp.to_csv(results_path+"\%s\linear_regression\hypothesis_test_std_residuals.csv"%scale)
    # Métricas de Erro
    df_metrics = pd.DataFrame({
        "MSE":  [mean_squared_error(y, fitted_model.fittedvalues)],
        "RMSE": [np.sqrt(mean_squared_error(y, fitted_model.fittedvalues))],
        "MAPE": [mean_absolute_percentage_error(y, fitted_model.fittedvalues)],
        "MAE":  [mean_absolute_error(y, fitted_model.fittedvalues)],
        "R2": [fitted_model.rsquared],
        "AdjustedR2": [fitted_model.rsquared_adj],
        "AIC":[fitted_model.aic],
        "BIC":[fitted_model.bic],
        
    })
    df_metrics.to_csv(results_path+"\%s\linear_regression\error_metrics.csv"%scale)


############### Gráficos dos resíduos ####################

fitted_model = fitted_models[0]
plot_acf(fitted_model.resid)
plt.ylim(-0.2, 0.2)
plt.title("Autocorrelação dos resíduos padronizados")
plt.xlabel('Lags')
plt.ylabel('FAC')
plt.show()

for (i, fitted_model) in enumerate(fitted_models):
    print(i)
    residuals = fitted_model.resid
    std_residuals = fitted_model.resid_pearson
    scale = scales[i]
    # FAC
    plt.figure(figsize=(10,5))
    plot_acf(std_residuals)
    plt.ylim(-0.15, 0.15)
    plt.title("Autocorrelação dos resíduos padronizados")
    plt.xlabel('Lags')
    plt.ylabel('FAC')
    plt.savefig(results_path+"\%s\linear_regression\\fac_std_residuals.jpg"%scale)
    plt.close()
    # Resíduos
    plt.figure(figsize=(10,5))
    plt.plot(std_residuals)
    plt.title("Resíduos padronizados")
    plt.savefig(results_path+"\%s\linear_regression\std_residuals.jpg"%scale)
    plt.close()
    # Histograma
    plt.figure(figsize=(10,8))
    sns.histplot(std_residuals)
    plt.title("Histograma dos resíduos padronizados")
    plt.savefig(results_path+"\%s\linear_regression\histogram_std_residuals.jpg"%scale)
    plt.close()
    # QQPlot
    plt.figure(figsize=(10,5))
    sm.qqplot(std_residuals, line="45")
    plt.xlim(-6, 6)
    plt.title("QQPlot dos resíduos padronizado")
    plt.savefig(results_path+"\%s\linear_regression\qqplot_std_residuals.jpg"%scale)
    plt.close()
    


############### Teste F para apenas as variáveis do GT ####################


for (i, fitted_model) in enumerate(fitted_models):
    scale = scales[i]
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
    df_f_gt.to_csv(results_path+"\%s\linear_regression\\f_test_gt.csv"%scale)
