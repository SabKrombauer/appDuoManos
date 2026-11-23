"""API local para testes de reconhecimento do DuoManos.

Execute: python -m uvicorn api:app --host 0.0.0.0 --port 8000
Nao exponha este servidor diretamente na internet.
"""
from pathlib import Path
import os
from threading import Lock
import pickle

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_libras.pkl"
LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"
MAX_BYTES = 5 * 1024 * 1024

if not MODEL_PATH.exists() or not LANDMARKER_PATH.exists():
    raise RuntimeError(
        "Arquivos modelo_libras.pkl e hand_landmarker.task devem estar ao lado de api.py"
    )

# Arquivos pickle executam codigo ao carregar: use somente o modelo treinado por voce.
with MODEL_PATH.open("rb") as arquivo:
    modelo = pickle.load(arquivo)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)
detector_lock = Lock()

app = FastAPI(title="DuoManos - API local de Libras")
# Para desenvolvimento local: origens permitidas podem ser configuradas via CORS_ORIGINS.
origens = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origem.strip() for origem in origens if origem.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def normalizar_duas_maos(hand_landmarks, handedness_list):
    """Mesma ordem e normalizacao usadas em coleta_libras.py."""
    mao_esq = np.zeros((21, 3))
    mao_dir = np.zeros((21, 3))
    for idx, mao in enumerate(hand_landmarks):
        lado = handedness_list[idx][0].category_name
        pontos = np.array([[p.x, p.y, p.z] for p in mao])
        pontos -= pontos[0]
        escala = np.max(np.abs(pontos))
        if escala > 0:
            pontos /= escala
        if lado == "Left":
            mao_esq = pontos
        else:
            mao_dir = pontos
    return np.concatenate((mao_esq.flatten(), mao_dir.flatten()))


@app.get("/saude")
def saude():
    return {"status": "ok"}


@app.post("/reconhecer")
async def reconhecer(imagem: UploadFile = File(...)):
    if imagem.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Envie uma imagem JPEG ou PNG.")

    dados = await imagem.read(MAX_BYTES + 1)
    if len(dados) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Imagem maior que 5 MB.")
    if not dados:
        raise HTTPException(status_code=400, detail="Imagem vazia.")

    frame = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem invalida.")

    # A coleta e os testes existentes espelham o frame antes do MediaPipe.
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # O detector e compartilhado entre requisicoes; evitar acesso simultaneo.
    with detector_lock:
        resultado = detector.detect(imagem_mp)
        if not resultado.hand_landmarks:
            return {"sinal": None, "confianca": 0.0, "maos_detectadas": 0}

        vetor = normalizar_duas_maos(
            resultado.hand_landmarks, resultado.handedness
        ).reshape(1, -1)
        sinal = str(modelo.predict(vetor)[0])
        confianca = float(np.max(modelo.predict_proba(vetor)[0]))

    if sinal.lower() == "desconhecido":
        sinal = None
    return {
        "sinal": sinal,
        "confianca": round(confianca, 4),
        "maos_detectadas": len(resultado.hand_landmarks),
    }
