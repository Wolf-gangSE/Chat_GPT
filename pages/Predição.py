import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

st.title('Predição de Qualidade de Vinhos')

# Solicitar ao usuário os valores das características do vinho
fixed_acidity = st.slider('Acidez Fixa', 3.0, 15.0, 8.0, 0.1)
volatile_acidity = st.slider('Acidez Volátil', 0.0, 2.0, 1.0, 0.1)
citric_acid = st.slider('Ácido Cítrico', 0.0, 1.0, 0.5, 0.1)
residual_sugar = st.slider('Açúcar Residual', 0.0, 15.0, 8.0, 0.1)
chlorides = st.slider('Cloretos', 0.0, 1.5, 0.5, 0.1)
free_sulfur_dioxide = st.slider('Dióxido de Enxofre Livre', 1, 100, 50, 1)
total_sulfur_dioxide = st.slider('Dióxido de Enxofre Total', 6, 300, 150, 1)
density = st.slider('Densidade', 0.9, 1.5, 1.0, 0.001)
pH = st.slider('pH', 2.5, 4.0, 3.0, 0.1)
sulphates = st.slider('Sulfatos', 0.0, 2.0, 1.0, 0.1)
alcohol = st.slider('Álcool', 8.0, 16.0, 10.0, 0.1)

# Leitura do arquivo CSV
path = 'WineQT.csv'
df = pd.read_csv(path)

# Separar as colunas de entrada (X) e saída (y)
X = df.drop(['qualidade', 'qualidade_bin', 'Id'], axis=1)
y = df['qualidade_bin']

# Padronizar as colunas de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Treinar o modelo Logistic Regression
lr = LogisticRegression(random_state=0)
lr.fit(X, y)

# Realizar a predição
prediction = lr.predict(scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, 
                                            sulphates, alcohol]]))

# Exibir o resultado da predição
st.subheader('Resultado da Predição')
if prediction[0] == 0:
    st.error('A qualidade do vinho é inferior.')
else:
    st.success('A qualidade do vinho é superior.')