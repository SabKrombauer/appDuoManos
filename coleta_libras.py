import csv
import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Configurado para detectar ATÉ 2 MÃOS
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

ARQUIVO = "dados.csv"


def normalizar_duas_maos(hand_landmarks, handedness_list):
    """
    Gera sempre 126 coordenadas (63 mão esquerda + 63 mão direita).
    Se apenas 1 mão estiver visível, preenche a outra com zeros.
    """
    mao_esq = np.zeros((21, 3))
    mao_dir = np.zeros((21, 3))

    for idx, mao in enumerate(hand_landmarks):
        lado = handedness_list[idx][0].category_name  # "Left" ou "Right"
        
        pts = np.array([[p.x, p.y, p.z] for p in mao])
        pts -= pts[0]  # Pulso como origem
        escala = np.max(np.abs(pts))
        if escala > 0:
            pts /= escala

        if lado == "Left":
            mao_esq = pts
        else:
            mao_dir = pts

    return np.concatenate([mao_esq.flatten(), mao_dir.flatten()])


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

print("DICA: Além dos sinais (oi, trabalhar), grave também o sinal 'desconhecido'")
print("fazendo gestos aleatórios ou mão em repouso!")
sinal = input("Nome do sinal que vai gravar: ").strip().lower()

cap = cv2.VideoCapture(0)
total = 0

with open(ARQUIVO, "a", newline="", encoding="utf-8") as arquivo:
    escritor = csv.writer(arquivo)

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        resultados = detector.detect(mp_image)
        mao_detectada = bool(resultados.hand_landmarks)

        if mao_detectada:
            h, w, _ = frame.shape
            for mao in resultados.hand_landmarks:
                for p1, p2 in HAND_CONNECTIONS:
                    pt1 = (int(mao[p1].x * w), int(mao[p1].y * h))
                    pt2 = (int(mao[p2].x * w), int(mao[p2].y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                for p in mao:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("s") and mao_detectada:
            vetor = normalizar_duas_maos(resultados.hand_landmarks, resultados.handedness)
            escritor.writerow([sinal, *vetor])
            total += 1

        cv2.putText(
            frame,
            f"Sinal: {sinal} | Gravados: {total} | S = gravar | Q = sair",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if mao_detectada else (0, 0, 255),
            2,
        )
        cv2.imshow("Coleta de Sinais (1 ou 2 maos)", frame)

        if tecla == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()