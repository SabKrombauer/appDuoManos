"""Janela de reconhecimento temporal para modelo_sequencias.pkl.

Uma amostra de treino tem 24 quadros x 132 atributos, a 12 FPS.
Este adaptador reamostra leituras do navegador e usa diferencas entre quadros.
Dados da webcam ficam apenas na memoria por sessao e nao sao gravados.
"""
from collections import deque
from threading import Lock
import re
import time

import numpy as np

STEPS = 24
FPS_TREINO = 12
JANELA_S = (STEPS - 1) / FPS_TREINO
PRESENCA = (65, 131)


class ReconhecedorTemporal:
    def __init__(self, modelo, confianca_min=0.78):
        self.modelo = modelo
        self.confianca_min = confianca_min
        self.historicos = {}
        self.ultimo_acesso = {}
        self.lock = Lock()

    def adicionar(self, sessao, pontos, permitir_parado=False):
        """Devolve (resultado, progresso 0..24).

        resultado e None quando nao existe janela/movimento/confianca suficiente.
        """
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", sessao):
            return None, 0
        pontos = np.asarray(pontos, dtype=np.float32)
        if pontos.shape != (132,) or not np.all(np.isfinite(pontos)):
            return None, 0
        agora = time.monotonic()

        with self.lock:
            # Nao mantenha dados de sessoes antigas na memoria indefinidamente.
            for chave in list(self.ultimo_acesso):
                if agora - self.ultimo_acesso[chave] > 15:
                    self.historicos.pop(chave, None)
                    self.ultimo_acesso.pop(chave, None)
            if sessao not in self.historicos:
                # Numero de sessoes limitado para este servidor local.
                if len(self.historicos) >= 8:
                    antiga = min(self.ultimo_acesso, key=self.ultimo_acesso.get)
                    self.historicos.pop(antiga, None)
                    self.ultimo_acesso.pop(antiga, None)
                self.historicos[sessao] = deque(maxlen=48)
            historico = self.historicos[sessao]
            self.ultimo_acesso[sessao] = agora

            if historico and agora - historico[-1][0] > 0.45:
                historico.clear()
            if historico and agora - historico[-1][0] < 0.05:
                return None, min(STEPS, len(historico))
            historico.append((agora, pontos.copy()))

            if len(historico) < 4 or agora - historico[0][0] < JANELA_S:
                duracao = agora - historico[0][0] if historico else 0
                return None, min(STEPS - 1, round(duracao * FPS_TREINO) + 1)

            timestamps = np.array([t for t, _ in historico], dtype=np.float64)
            valores = np.stack([p for _, p in historico], axis=0)
            # As ultimas ~2 s sao interpoladas para 24 quadros fixos, exatamente
            # como o formato de treino. Periodos sem dados reiniciam a janela.
            if np.max(np.diff(timestamps)) > 0.45:
                historico.clear()
                return None, 0
            instantes = np.linspace(agora - JANELA_S, agora, STEPS)
            frames = np.empty((STEPS, 132), dtype=np.float32)
            for indice in range(132):
                frames[:, indice] = np.interp(instantes, timestamps, valores[:, indice])
            # Presenca de mao deve continuar binaria.
            frames[:, list(PRESENCA)] = (frames[:, list(PRESENCA)] >= 0.5).astype(np.float32)

            if np.count_nonzero(np.any(frames[:, list(PRESENCA)] > 0, axis=1)) < STEPS // 2:
                return None, STEPS

            # So prioriza o modelo temporal quando existe movimento suficiente.
            # Isso reduz falsos positivos ao manter sinais estaticos.
            pulso_esq = frames[:, 63:65]
            pulso_dir = frames[:, 129:131]
            movimento = (
                np.sum(np.linalg.norm(np.diff(pulso_esq, axis=0), axis=1))
                + np.sum(np.linalg.norm(np.diff(pulso_dir, axis=0), axis=1))
            )
            if movimento < 0.12 and not permitir_parado:
                return None, STEPS

            deltas = np.diff(frames, axis=0, prepend=frames[:1])
            entrada = np.concatenate([frames, deltas], axis=1).reshape(1, -1)
            probabilidades = self.modelo.predict_proba(entrada)[0]
            melhor = int(np.argmax(probabilidades))
            rotulo = str(self.modelo.classes_[melhor])
            confianca = float(probabilidades[melhor])
            if rotulo.lower() == "desconhecido" or confianca < self.confianca_min:
                return None, STEPS
            return {"sinal": rotulo, "confianca": round(confianca, 4)}, STEPS
