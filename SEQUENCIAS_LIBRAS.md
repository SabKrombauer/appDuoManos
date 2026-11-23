# Etapa 2: reconhecer sinais com movimento (duas mãos)

O modelo atual (`modelo_libras.pkl`) classifica **uma imagem por vez**.
Por isso pode reconhecer EU TE AMO parado e confundir OI durante o movimento.
O MediaPipe já recebe até duas mãos, mas faltam exemplos temporais.

## Coleta nova (não substitui os arquivos atuais)

Na raiz do DuoManos (onde está `hand_landmarker.task`):

```powershell
python coletar_sequencias.py
```

1. Informe o sinal (por exemplo `oi`).
2. Com as mãos visíveis na webcam, pressione **Espaço** e realize **o sinal inteiro**, do começo ao fim.
3. A gravação captura 24 leituras, aproximadamente 2 s, a 12 leituras por segundo.
4. Repita várias vezes, variando velocidade e posição, e use **Q** para encerrar.
5. Execute novamente para `eu te amo`, `trabalhar`, `desconhecido` e novos sinais.

A coleta guarda, **por quadro**, 21 pontos por mão, mais a posição
absoluta dos pulsos e um indicador de presença. São **132 valores por quadro**
(66 por mão). Esse formato mantém a trajetória entre quadros, ao contrário
da normalização estática de `dados.csv`. Sinais de duas mãos devem ser
gravados com as duas mãos visíveis do começo ao fim.

As amostras são armazenadas localmente em `sequencias/<sinal>/*.npz`.
Os arquivos estão no `.gitignore`: não envie amostras pessoais ao GitHub
sem autorização explícita de quem aparece nas gravações.

## Treinamento experimental

Depois de gravar **20-40 execuções completas por classe**, preferencialmente
com pessoas diferentes, execute:

```powershell
python treinar_sequencias.py
```

Isso cria `modelo_sequencias.pkl` e `modelo_sequencias_info.json`.
O treinamento usa Random Forest com pontos e deslocamentos entre quadros;
é uma primeira base temporal mais simples que LSTM.

**Atenção:** os exemplos precisam representar cada execução inteira do sinal.
O teste aleatório impresso pelo script não prova generalização; separe pessoas
ou sessões diferentes para validar e conte falsos reconhecimentos. É
importante validar os sinais e o vocabulário com pessoas fluentes em Libras.

## O que ainda falta para uso no Sinaliza

Treinar um modelo temporal não o conecta automaticamente ao aplicativo.
A integração em tempo real precisará receber uma janela de frames ordenados
pelo tempo e usar `modelo_sequencias.pkl` no endpoint Python, substituindo
ou combinando o modelo estático. **Ainda não foi implementada a inferência
temporal no Sinaliza.** Os arquivos novos são a preparação da coleta e do
treinamento, sem alterar o reconhecimento que já funciona.
