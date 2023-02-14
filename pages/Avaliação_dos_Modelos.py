import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
import pickle

# Leitura do arquivo CSV
url = 'https://raw.githubusercontent.com/Wolf-gangSE/databases/main/WineQT.csv'
df = pd.read_csv(url)

# Renomeação das colunas
df = df.rename(columns={'fixed acidity': 'acidez fixa', 'volatile acidity': 'acidez volatil',
                        'citric acid': 'acido citrico', 'residual sugar': 'açucar residual',
                        'chlorides': 'cloreto', 'free sulfur dioxide': 'dioxido de enxofre livre',
                        'total sulfur dioxide': 'dioxido de enxofre total', 'density': 'densidade',
                        'pH': 'pH', 'sulphates': 'sulfatos', 'alcohol': 'alcool',
                        'quality': 'qualidade'})

# Criação da coluna "qualidade_bin"
df['qualidade_bin'] = df['qualidade'].apply(lambda x: 0 if int(x) <= 5 else 1)

df.to_csv('WineQT.csv', index=False)

# Selecionando features 
df_features = df.drop(['qualidade', 'qualidade_bin', 'Id'], axis=1)

# Selecionar apenas as colunas numéricas
numeric_columns = df_features.select_dtypes(include=['int64', 'float64'])

# Obter o array com os nomes das colunas numéricas
numeric_features = numeric_columns.columns.values
print(numeric_features)

# Seelcionar apenas as colunas categóricas
categorical_columns = df_features.select_dtypes(include=['object'])

# Obter o array com os nomes das colunas categóricas
categorical_features = categorical_columns.columns.values
print(categorical_features)

# Definição das transformações para cada tipo de atributo
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar as transformações em uma única transformação
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])



# Definição dos modelos a serem avaliados
models = [DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier()]

# Definição dos parâmetros a serem testados para cada modelo
params = [{'clf__criterion': ['gini', 'entropy'], 'clf__max_depth': [2, 4, 6, 8, 10]},
          {'clf__penalty': ['l2'], 'clf__C': [0.001, 0.01, 0.1, 1, 10]},
          {'clf__n_neighbors': [1, 3, 5, 7, 9], 'clf__weights': ['uniform', 'distance'], 'clf__p': [1, 2]}]

# Definição da pipeline para pré-processamento e classificação
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', None)])

# Avaliação dos modelos utilizando Cross Validation e Grid Search
results = []

st.write('# Avaliação dos Modelos')

tabs = st.tabs(['Decision Tree', 'Logistic Regression', 'KNN'])

print(tabs)

for model, param, tab in zip(models, params,tabs):
    with tab:
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', model)])

        st.info(f'Modelo Utilizado: {type(model).__name__}')
        grid_search = GridSearchCV(pipe, param, cv=5, scoring='accuracy')
        grid_search.fit(df.drop(['qualidade', 'qualidade_bin'], axis=1), df['qualidade_bin'])
        
        cv_results = cross_validate(grid_search.best_estimator_, df.drop(['qualidade', 'qualidade_bin'], axis=1), df['qualidade_bin'], cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], )
        
        results.append({
            'model': type(model).__name__,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': cv_results
        })
 
        # salvar o modelo XGBoost (xgb_model) no arquivo sale_xgboost.pkl
        with open(f'models/{type(model).__name__}.pkl', 'wb') as file:
            pickle.dump(grid_search.best_estimator_, file)

        
        st.write('## Resultados')
        st.metric('Melhores Parâmetros', str(grid_search.best_params_))

        st.metric('Melhor Score', f'{grid_search.best_score_:.2f}')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Accuracy', f'{cv_results["test_accuracy"].mean():.2f} ({cv_results["test_accuracy"].std():.2f})')
            st.metric('Precision', f'{cv_results["test_precision"].mean():.2f} ({cv_results["test_precision"].std():.2f})')
        with col2:
            st.metric('Recall', f'{cv_results["test_recall"].mean():.2f} ({cv_results["test_recall"].std():.2f})')
            st.metric('F1-Score', f'{cv_results["test_f1"].mean():.2f} ({cv_results["test_f1"].std():.2f})')