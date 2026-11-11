import cv2
import numpy as np
import os
import pickle
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Carrega o modelo de IA treinado
if not os.path.exists("modelo_libras.pkl"):
    print("Erro: 'modelo_libras.pkl' não encontrado. Rode o treinar_modelo.py primeiro!")
    exit()

with open("modelo_libras.pkl", "rb") as f:
    modelo = pickle.load(f)

# 2. Configura o MediaPipe
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)


def normalizar_duas_maos(hand_landmarks, handedness_list):
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


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Configurações do Formador de Frases
LIMITE_CONFIANCA = 0.82
FRAMES_NECESSARIOS = 10  # Quantos frames (~0.3s) o sinal precisa ficar firme para ser aceito

frase = []
palavra_candidata = ""
contador_frames = 0
pode_adicionar = True

cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    resultados = detector.detect(mp_image)
    sinal_detectado_no_frame = False

    if resultados.hand_landmarks:
        h, w, _ = frame.shape
        for mao in resultados.hand_landmarks:
            for p1, p2 in HAND_CONNECTIONS:
                pt1 = (int(mao[p1].x * w), int(mao[p1].y * h))
                pt2 = (int(mao[p2].x * w), int(mao[p2].y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            for p in mao:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)

        vetor = normalizar_duas_maos(resultados.hand_landmarks, resultados.handedness).reshape(1, -1)
        predicao = modelo.predict(vetor)[0]
        probabilidades = modelo.predict_proba(vetor)
        confianca = np.max(probabilidades)

        if confianca >= LIMITE_CONFIANCA and predicao != "desconhecido":
            sinal_detectado_no_frame = True

            # Conta se o mesmo sinal está sendo mantido
            if predicao == palavra_candidata:
                contador_frames += 1
            else:
                palavra_candidata = predicao
                contador_frames = 0

            # Adiciona a palavra à frase se atingiu a estabilidade e a trava permitir
            if contador_frames >= FRAMES_NECESSARIOS and pode_adicionar:
                frase.append(predicao.upper())
                pode_adicionar = False  # Trava para não repetir a mesma palavra
                print(f"Palavra adicionada: {predicao.upper()}")

    # Se a mão saiu da tela ou fez um gesto não reconhecido, reseta a trava
    if not sinal_detectado_no_frame:
        palavra_candidata = ""
        contador_frames = 0
        pode_adicionar = True

    # --- DESENHO DA INTERFACE NA TELA ---
    
    # Faixa preta no topo para exibir a frase acumulada
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    
    texto_frase = " ".join(frase)
    if not texto_frase:
        texto_frase = "Faça os sinais para formar a frase..."

    # Exibe a frase na tela
    cv2.putText(
        frame,
        texto_frase,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Dicas de comandos no rodapé da tela
    cv2.putText(
        frame,
        "Controles: [C] Limpar Frase | [B] Apagar Ultima Palavra | [Q] Sair",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.imshow("Tradutor de Libras em Frases", frame)

    tecla = cv2.waitKey(1) & 0xFF
    
    # Tecla C: limpa toda a frase
    if tecla == ord("c"):
        frase.clear()
    
    # Tecla B: apaga a última palavra da frase
    if tecla == ord("b") and len(frase) > 0:
        frase.pop()

    # Tecla Q: sai do aplicativo
    if tecla == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()