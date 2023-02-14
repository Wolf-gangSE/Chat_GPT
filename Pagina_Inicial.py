import streamlit as st
import pandas as pd
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="📈 Página inicial", page_icon="📈", layout='wide')
st.markdown('''
<style>
  section.main > div {max-width:80em}
</style>
''', unsafe_allow_html=True)

# Leitura do arquivo CSV
path = 'WineQT.csv'
df = pd.read_csv(path)

st.header('Análise do Dataset WineQT')


st.markdown("""---""")

# Gráfico de barras para visualizar a distribuição das classes
st.subheader('Distribuição das classes:')
fig = px.histogram(df, x='qualidade_bin', nbins=2)
fig.update_layout(xaxis_title='Qualidade', yaxis_title='Frequência', bargap=0.2)
st.plotly_chart(fig, use_container_width=True)


st.markdown("""---""")

st.subheader('Dispersão entre as variáveis:')

# Selecionando features 
df_features = df.drop(['qualidade', 'qualidade_bin', 'Id'], axis=1)

# Selecionar apenas as colunas numéricas
numeric_columns = df_features.select_dtypes(include=['int64', 'float64'])

# Obter o array com os nomes das colunas numéricas
numeric_features = numeric_columns.columns

selected1 = st.selectbox('Selecione uma variável:', numeric_features.tolist())

if selected1 in numeric_features.tolist():
  selected2 = st.selectbox('Selecione outra variável:', numeric_features.tolist())
else:
  selected2 = st.selectbox('Selecione outra variável:', numeric_features.tolist())
# Gráfico de dispersão para visualizar a relação entre duas variáveis
fig2 = px.scatter(df, x=selected1, y=selected2)
st.plotly_chart(fig2, use_container_width=True)


st.markdown("""---""")
st.subheader('Gráfico de caixas por classificação:')
# Gráfico de caixas para visualizar a distribuição de uma variável em cada classe
fig3 = px.box(df, x='qualidade_bin', y='alcool')
st.plotly_chart(fig3, use_container_width=True)


st.markdown("""---""")
st.subheader('Matriz de correlação:')
# Matriz de correlação para visualizar a relação entre as variáveis
corr = df.corr()
fig4 = px.imshow(corr)
st.plotly_chart(fig4, use_container_width=True)
