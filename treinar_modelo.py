import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = [], []

with open("dados.csv", "r", encoding="utf-8") as f:
    leitor = csv.reader(f)
    for linha in leitor:
        if linha:
            y.append(linha[0])
            X.append([float(val) for val in linha[1:]])

X = np.array(X)
y = np.array(y)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Treinando o modelo de Machine Learning...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
print(f"Precisão do modelo nos testes: {acuracia * 100:.2f}%")

with open("modelo_libras.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Modelo salvo em 'modelo_libras.pkl'!")