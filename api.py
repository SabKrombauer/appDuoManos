"""API local do DuoManos. Executar somente com modelo treinado por você."""
from __future__ import annotations
import base64
import binascii
import os
import pickle
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "modelo_libras.pkl"
LANDMARKER_PATH = BASE / "hand_landmarker.task"

if not MODEL_PATH.is_file():
    raise RuntimeError("modelo_libras.pkl não encontrado. Execute treinar_modelo.py antes.")
if not LANDMARKER_PATH.is_file():
    raise RuntimeError("hand_landmarker.task não encontrado. Execute coleta_libras.py antes.")

# ATENÇÃO: arquivos pickle podem executar código. Carregue somente o modelo que você treinou.
with MODEL_PATH.open("rb") as arquivo:
    modelo = pickle.load(arquivo)

opcoes = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(opcoes)
lock = Lock()
app = FastAPI(title="DuoManos — Reconhecimento local")


class ImagemEntrada(BaseModel):
    imagem: str = Field(min_length=10, max_length=6_000_000)


def normalizar_duas_maos(hand_landmarks, handedness_list):
    """MESMA normalização usada em coleta_libras.py e testar_modelo.py."""
    mao_esq = np.zeros((21, 3))
    mao_dir = np.zeros((21, 3))
    for idx, mao in enumerate(hand_landmarks):
        lado = handedness_list[idx][0].category_name
        pts = np.array([[p.x, p.y, p.z] for p in mao])
        pts -= pts[0]
        escala = np.max(np.abs(pts))
        if escala > 0:
            pts /= escala
        if lado == "Left":
            mao_esq = pts
        else:
            mao_dir = pts
    return np.concatenate([mao_esq.flatten(), mao_dir.flatten()])


@app.get("/saude")
def saude():
    return {"status": "ok"}


@app.post("/prever")
def prever(entrada: ImagemEntrada):
    try:
        dados = base64.b64decode(entrada.imagem, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Base64 inválido.")
    if len(dados) > 4_500_000:
        raise HTTPException(status_code=413, detail="Imagem muito grande.")

    buffer = np.frombuffer(dados, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem inválida.")
    # A webcam de testar_modelo.py era espelhada.
    # Caso o celular já envie selfie espelhada, remova esta linha.
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with lock:
        resultado = detector.detect(imagem)
        quantidade = len(resultado.hand_landmarks)
        if quantidade == 0:
            return {"sinal": None, "confianca": 0.0, "maos_detectadas": 0}
        vetor = normalizar_duas_maos(
            resultado.hand_landmarks, resultado.handedness
        ).reshape(1, -1)
        predicao = str(modelo.predict(vetor)[0])
        confianca = float(np.max(modelo.predict_proba(vetor)[0]))

    return {
        "sinal": predicao,
        "confianca": round(confianca, 4),
        "maos_detectadas": quantidade,
    }
