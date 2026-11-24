"""Testes unitarios da janela temporal; nao usa webcam nem modelos reais."""
import unittest
from unittest.mock import patch

import numpy as np
from reconhecedor_temporal import ReconhecedorTemporal


class ModeloFalso:
    classes_ = np.array(["oi", "trabalhar"])

    def __init__(self):
        self.ultima_entrada = None

    def predict_proba(self, entrada):
        self.ultima_entrada = entrada
        return np.array([[0.92, 0.08]], dtype=float)


class TesteReconhecedorTemporal(unittest.TestCase):
    def test_reconhece_movimento_depois_de_dois_segundos(self):
        modelo = ModeloFalso()
        reconhecedor = ReconhecedorTemporal(modelo)
        resultados = []
        for i in range(28):
            pontos = np.zeros(132, dtype=np.float32)
            pontos[65] = 1.0
            pontos[63] = 0.4 + 0.13 * np.sin(i / 3)
            with patch("reconhecedor_temporal.time.monotonic", return_value=100 + i / 12):
                resultados.append(reconhecedor.adicionar("camera-1", pontos))
        self.assertEqual(resultados[-1][0]["sinal"], "oi")
        self.assertEqual(modelo.ultima_entrada.shape, (1, 24 * 132 * 2))

    def test_pose_parada_nao_dispara_modelo_temporal(self):
        modelo = ModeloFalso()
        reconhecedor = ReconhecedorTemporal(modelo)
        for i in range(28):
            pontos = np.zeros(132, dtype=np.float32)
            pontos[65] = 1.0
            pontos[63] = 0.4
            with patch("reconhecedor_temporal.time.monotonic", return_value=100 + i / 12):
                resultado, _ = reconhecedor.adicionar("camera-1", pontos)
        self.assertIsNone(resultado)
        self.assertIsNone(modelo.ultima_entrada)

    def test_duas_maos_estaticas_podem_ser_classificadas_quando_preciso(self):
        modelo = ModeloFalso()
        reconhecedor = ReconhecedorTemporal(modelo)
        for i in range(28):
            pontos = np.zeros(132, dtype=np.float32)
            pontos[65] = 1.0
            pontos[131] = 1.0
            pontos[63] = 0.4
            pontos[129] = 0.65
            with patch("reconhecedor_temporal.time.monotonic", return_value=100 + i / 12):
                resultado, _ = reconhecedor.adicionar(
                    "camera-2", pontos, permitir_parado=True
                )
        self.assertEqual(resultado["sinal"], "oi")

    def test_sessoes_invalidas_sao_rejeitadas(self):
        resultado, progresso = ReconhecedorTemporal(ModeloFalso()).adicionar(
            "uma sessao com espacos", np.zeros(132, dtype=np.float32)
        )
        self.assertIsNone(resultado)
        self.assertEqual(progresso, 0)


if __name__ == "__main__":
    unittest.main()
