"""Treina classificador temporal experimental a partir de sequencias/*.npz.

Rode depois de coletar pelo menos 20 execucoes por classe, de preferencia
de pessoas diferentes. Esse modelo NAO substitui modelo_libras.pkl.
"""
from pathlib import Path
from collections import Counter
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
PASTA = BASE / "sequencias"
X, y = [], []

for arquivo in sorted(PASTA.glob("*/*.npz")):
    with np.load(arquivo, allow_pickle=False) as amostra:
        frames = amostra["frames"]
        rotulo = str(amostra["rotulo"].item())
    if frames.shape != (24, 132) or not np.all(np.isfinite(frames)):
        print(f"Ignorando arquivo incorreto: {arquivo}")
        continue
    # Posições e deslocamentos entre os quadros: preserva a ordem temporal.
    velocidades = np.diff(frames, axis=0, prepend=frames[:1])
    caracteristicas = np.concatenate([frames, velocidades], axis=1).flatten()
    X.append(caracteristicas)
    y.append(rotulo)

contagem = Counter(y)
print("Amostras por sinal:", dict(contagem))

# Classes ainda com poucos exemplos ficam guardadas em sequencias/ para o
# proximo treinamento. Nao as inclua como classes com poucos dados: elas
# impediam o primeiro teste, mesmo com outros sinais suficientes.
minimo = 10
incluidas = {rotulo for rotulo, total in contagem.items() if total >= minimo}
ignoradas = {rotulo: total for rotulo, total in contagem.items() if total < minimo}
if ignoradas:
    print("Sinais com menos de 10 gravacoes (serao ignorados neste treino):", ignoradas)
if len(incluidas) < 2:
    raise SystemExit(
        "Ainda nao ha duas classes com pelo menos 10 gravacoes validas cada. "
        "Grave mais amostras antes de treinar."
    )

indices = [i for i, rotulo in enumerate(y) if rotulo in incluidas]
X = np.asarray([X[i] for i in indices], dtype=np.float32)
y = np.asarray([y[i] for i in indices])
contagem_treino = Counter(y)
print("Sinais incluidos no treinamento:", dict(contagem_treino))
print("Total de sequencias utilizadas:", len(y))
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)
modelo = RandomForestClassifier(
    n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
modelo.fit(X_treino, y_treino)
print(classification_report(y_teste, modelo.predict(X_teste), zero_division=0))
print("ATENCAO: teste aleatorio superestima o desempenho em pessoas nao vistas.")
print("Valide em gravacoes de outra pessoa antes de usar no aplicativo.")

saida = BASE / "modelo_sequencias.pkl"
with saida.open("wb") as arquivo:
    pickle.dump(modelo, arquivo)
(BASE / "modelo_sequencias_info.json").write_text(
    json.dumps({"classes": sorted(contagem_treino), "frames": 24, "features_per_frame": 132},
               ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Modelo experimental salvo:", saida)
