# **Setup**

Instalação das bibliotecas
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install plotly
# %pip install cufflinks
# %pip install chart-studio

"""Importação das principais bilbiotecas ultilizadas"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import cufflinks as cf

import plotly.graph_objects as go
import plotly.express as px

MKT_Dados = pd.read_csv("/content/drive/MyDrive/Colab - Desafio 4/MKT.csv")

"""# **Análise Descritiva**

Visualização dos dados da empresa
"""

MKT_Dados.head()

"""Visualização dos tipos de dados"""

MKT_Dados.dtypes

"""Alteração dos Tipos de dados"""

MKT_Dados = MKT_Dados.astype({"youtube":"int", "facebook":"int", "newspaper":"int", "sales":"int"})
MKT_Dados.dtypes

"""Visualização das informações dos dados"""

MKT_Dados.info()

"""Ultilização da função "describe"
"""

MKT_Dados[["youtube","facebook","newspaper","sales"]].describe()

"""# **Análise Exploratória**

Grafico de correlação Youtube X Vendas
"""

ax = sns.barplot(data=MKT_Dados,x="sales",y="youtube");

"""Grafico de correlação Facebook X Vendas"""

ax = sns.barplot(data=MKT_Dados,x="sales",y="facebook");

"""Grafica de correlação Newspaper X Vendas"""

ax = sns.barplot(data=MKT_Dados,x="sales",y="newspaper");

"""Grafica de correlação Youtube, Facebook e Newspaper X Vendas"""

df = pd.DataFrame(MKT_Dados)

df_long = pd.melt(df, id_vars=['sales'], value_vars=['newspaper', 'facebook', 'youtube'],
                  var_name='channel', value_name='value')

sns.catplot(data=df_long, x='sales', y='value', hue='channel', kind='bar', height=6, aspect=2)

agg_sales = MKT_Dados[["sales"]]

agg_sales.describe()

"""Histograma de Vendas"""

sns.histplot(data = MKT_Dados, x = "sales");

"""Grafico de correlação detalhado usando fig Youtube, Facebook, Newspaper X Vendas"""

df = pd.DataFrame(MKT_Dados)

df_long = pd.melt(df, id_vars=['sales'], value_vars=['newspaper', 'facebook', 'youtube'],
                  var_name='channel', value_name='value')

fig = px.bar(df_long, x='sales', y='value', color='channel', barmode='group')

fig.show()

"""# **Modelagem**

Gráfico de Regreção
"""

fig = px.box(MKT_Dados, x = "sales")
fig.update_traces(line_color="blue")

"""Gráficos de Correlação da base de dados"""

sns.pairplot(MKT_Dados)

"""Graficos de Comparação em relação as Vendas"""

MKT_Dados.columns

sns.pairplot(MKT_Dados, x_vars=['youtube', 'facebook', 'newspaper'], y_vars="sales")

"""Correlação dos Dados"""

MKT_Dados.corr()

"""Correlação em Gráfico"""

sns.heatmap(MKT_Dados.corr(), annot =True)

"""Distribuição das Vendas"""

sns.histplot(MKT_Dados["sales"])

"""Tamanho dos Dados por Coluna"""

MKT_Dados.columns

x = MKT_Dados[['youtube', 'facebook', 'newspaper']]

y = MKT_Dados[["sales"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7,test_size = 0.3, random_state= 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

"""# **Calculando predição**"""

Margem de acerto da predição

"""Gráfico de Linha em Predição"""

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train, y_train)

y_pred = lm.predict(x_test)

from sklearn.metrics import r2_score
r = r2_score(y_test, y_pred)

print("r_quadrado:",r)

"""Gráfico de Linha em Predição"""

c = [i for i in range(1, 1501, 1)]
fig = plt.figure(figsize=(12,8))
plt.plot(c[:len(y_test)], y_test, color = "blue")
plt.plot(c[:len(y_pred)], y_pred, color = "red")
plt.xlabel("index")
plt.ylabel("vendas")

"""Predição Manual"""

youtube = 50
facebook = 30
newspaper = 7
entrada = [[youtube,facebook,newspaper]]
lm.predict(entrada)[0]
