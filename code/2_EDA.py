import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np

def line_plot(dates, y, title, xlabel, ylabel):
    p = plt.figure(figsize=(10, 6))
    p = plt.plot(dates, y, linestyle='-')
    p = plt.title(title, fontsize=14)
    p = plt.xlabel(xlabel, fontsize=12)
    p = plt.ylabel(ylabel, fontsize=12)
    p = plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    p = plt.xticks(fontsize=10)
    p = plt.yticks(fontsize=10)
    p = plt.gca().spines['top'].set_visible(False)
    p = plt.gca().spines['right'].set_visible(False)
    p = plt.tight_layout()
    return p

plt.style.use('seaborn-v0_8')

folder_path  = "Probest/TrabalhoFinal/Dados/"
results_path = "Probest/TrabalhoFinal/Resultados/"

target = "^BVSP"

scales = ["price_gt", "price_delta", "ret_gt", "ret_delta"]


df = pd.read_csv(folder_path + "\%s\%s_dataset.csv"%(scales[0], target))
df.date = pd.to_datetime(df.date)
cols = df.columns[2:].values
cols = np.delete(cols, 2)


for col in df.columns[2:]:
    p = line_plot(df.date, df[col], col, "tempo", "")
    plt.savefig(results_path+"/%s.png"%(col))


file_names = [results_path + f'{col}.png' for col in cols]
images = [mpimg.imread(file_name) for file_name in file_names]
# Criar a primeira figura com layout 1x3
fig1, axes1 = plt.subplots(4, 3, figsize=(15, 5))
for i, ax in enumerate(axes1):
    ax.imshow(images[i])
    ax.axis('off')  # Esconde os eixos
fig1.tight_layout()
fig1.suptitle('Gráficos das variáveis explicativas', fontsize=16)
fig1.savefig(results_path+"/explicativas.jpg")  # Salvar a primeira figura
    

for scale in scales:
    df = pd.read_csv(folder_path + "\%s\%s_dataset.csv"%(scale, target))
    correlation_target = df.iloc[:, 1:].corr()["^BVSP"]
    for nome in df.columns[2:]:
        x = df.loc[:,nome]
        y = df.loc[:,target]
        m, b = np.polyfit(x, y, 1)
        plt.figure(figsize=(8,8), dpi=200)
        plt.scatter(x, y, alpha=0.5, label = "")
        plt.plot(x, m*np.array(x) + b, color='red')
        plt.title("%s - (corr = %.4f)"%(nome,correlation_target[nome]), fontsize=20)
        plt.legend(loc = "upper left")
        plt.xlabel(nome, fontsize=16)
        plt.ylabel(target, fontsize=12)
        plt.tight_layout()
        plt.savefig(results_path+"%s/scatter/bvsp_%s.png"%(scale,nome))
        plt.close()


for scale in scales:
    df = pd.read_csv(folder_path + "\%s\%s_dataset.csv"%(scale, target))
    file_names = [results_path + f'{scale}/scatter/bvsp_{nome}.png' for nome in df.columns[2:]]
    # Ler as imagens
    images = [mpimg.imread(file_name) for file_name in file_names]
    # Criar a primeira figura com layout 1x3
    # fig1, axes1 = plt.subplots(1, 3, figsize=(12, 5))
    # for i, ax in enumerate(axes1):
    #     ax.imshow(images[i])
    #     ax.axis('off')  # Esconde os eixos
    # fig1.tight_layout()
    # fig1.suptitle('Gráficos de dispersão de ^BVSP X variáveis econômicas', fontsize=16)
    # fig1.savefig(results_path+"%s/scatter/economicas.jpg"%(scale))  # Salvar a primeira figura
    # Criar a segunda figura com layout 2x5
    fig2, axes2 = plt.subplots(4, 3, figsize=(10, 15))
    for i, ax in enumerate(axes2.flatten()):
        if i < len(images) - 3:
            ax.imshow(images[i + 3])
        ax.axis('off')  # Esconde os eixos
    fig2.subplots_adjust(wspace=0.1, hspace=0.9)
    fig2.suptitle('Gráficos de dispersão de ^BVSP X Google Trends', fontsize=16)
    fig2.savefig(results_path+"%s/scatter/g trends.jpg"%(scale))  # Salvar a segunda figura
    plt.close() 



