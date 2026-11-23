"""Coleta sinais com movimento usando MediaPipe e ate duas maos.

Execute na raiz do DuoManos: python coletar_sequencias.py
Tecla ESPACO grava uma sequencia de 24 quadros (~2 segundos).
Tecla Q encerra. Cada arquivo .npz equivale a uma execucao inteira do sinal.

Este formato e DIFERENTE de dados.csv e nao altera o modelo estatico.
"""
from pathlib import Path
from uuid import uuid4
import re
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "hand_landmarker.task"
DATA_DIR = BASE / "sequencias"
STEPS = 24
FPS_COLETA = 12
FEATURES = 132  # duas maos: (21*3 relativos + pulso XY absoluto + presente) * 2


def extrair_pontos(resultado):
    """Mantem ordem Left/Right como coleta_libras.py, agora incluindo pulso XY."""
    saida = np.zeros((2, 66), dtype=np.float32)
    for idx, mao in enumerate(resultado.hand_landmarks):
        lado = resultado.handedness[idx][0].category_name
        slot = 0 if lado == "Left" else 1
        pts = np.asarray([[p.x, p.y, p.z] for p in mao], dtype=np.float32)
        pulso = pts[0, :2].copy()
        relativo = pts - pts[0]
        escala = float(np.max(np.abs(relativo)))
        if escala > 1e-8:
            relativo /= escala
        saida[slot, :63] = relativo.flatten()
        saida[slot, 63:65] = pulso
        saida[slot, 65] = 1.0
    return saida.flatten()


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo MediaPipe nao encontrado: {MODEL_PATH}")
    tamanho = MODEL_PATH.stat().st_size
    if tamanho < 100_000:
        raise RuntimeError(
            f"hand_landmarker.task parece incompleto ({tamanho} bytes). "
            "Copie novamente o arquivo verdadeiro da pasta antiga do DuoManos."
        )
    rotulo = input("Nome do sinal (ex.: oi, trabalhar): ").strip().lower()
    if not rotulo:
        raise ValueError("Nome do sinal vazio")
    pasta = DATA_DIR / (re.sub(r"[^a-z0-9_-]+", "_", rotulo) or "sinal")
    pasta.mkdir(parents=True, exist_ok=True)
    print("Pressione ESPACO para gravar; Q para sair. Grave varias execucoes de cada sinal.")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam")

    detector = vision.HandLandmarker.create_from_options(options)
    gravando = False
    frames = []
    ultimo = 0.0
    total = len(list(pasta.glob("*.npz")))
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # igual ao pre-processamento atual
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            pontos = extrair_pontos(resultado)

            agora = time.monotonic()
            if gravando and agora - ultimo >= 1 / FPS_COLETA:
                frames.append(pontos)
                ultimo = agora
                if len(frames) == STEPS:
                    sequencia = np.stack(frames).astype(np.float32)
                    # Pelo menos metade da sequencia deve conter uma mao detectada.
                    if np.sum(np.any(sequencia[:, [65, 131]] > 0, axis=1)) >= STEPS // 2:
                        arquivo = pasta / f"{uuid4().hex}.npz"
                        np.savez_compressed(arquivo, frames=sequencia, rotulo=rotulo)
                        total += 1
                        print(f"Amostra {total} salva em: {arquivo}")
                    else:
                        print("Poucas maos detectadas; amostra descartada")
                    gravando = False

            texto = f"{rotulo} | salvas: {total} | ESPACO=gravar Q=sair"
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if gravando:
                cv2.putText(frame, f"GRAVANDO {len(frames)}/{STEPS}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 40, 255), 2)
            cv2.imshow("DuoManos - sequencias de Libras", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == 32 and not gravando:
                gravando = True
                frames = []
                ultimo = 0.0
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
