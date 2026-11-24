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

## Integracao experimental com Sinaliza

Esta branch inclui `reconhecedor_temporal.py` e atualiza a API `api.py`.
O classificador temporal e carregado **automaticamente ao iniciar o servidor**,
se `modelo_sequencias.pkl` existir ao lado de `api.py`.
O modelo estatico `modelo_libras.pkl` continua sendo usado como alternativa.

A API recebe `sessao` no mesmo POST `/reconhecer`, preserva ate aproximadamente
2 segundos de landmarks por sessao, reamostra 24 quadros (132 dados por quadro)
e aplica exatamente as caracteristicas do treinamento (posicoes + deslocamentos).
Para reduzir falsos positivos, so prioriza a previsao temporal quando as maos
se movimentaram o suficiente e a confianca do modelo for pelo menos 78%.

O `GET /saude` informa quais modelos estao carregados:

```json
{"status":"ok","modelo_estatico":true,"modelo_temporal":true}
```

Esse exemplo so mostrara ambos como `true` se ambos os arquivos `.pkl` estiverem
presentes. Reinicie a API depois de treinar para carregar o novo modelo.

### Teste local (Windows / VS Code)

Na pasta `DuoManos-git`, depois de atualizar para a branch
`feat/integracao-sequencias-ao-vivo`, execute:

```powershell
& "C:\Users\bakro\OneDrive\Documents\TESTE\.venv\Scripts\python.exe" ".\treinar_sequencias.py"
```

Se o treinamento terminou, mantenha `modelo_sequencias.pkl` na **mesma pasta**
de `api.py`. Para preservar o classificador estatico, copie também
`modelo_libras.pkl` da pasta antiga (se ainda nao estiver nessa pasta).
O `hand_landmarker.task` copiado durante a coleta deve continuar presente.

Pare a API anterior que usava a porta 8000 com `Ctrl + C` e inicie **nesta
nova pasta**:

```powershell
& "C:\Users\bakro\OneDrive\Documents\TESTE\.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
```

No Sinaliza, atualize o projeto para a branch do **mesmo nome** e mantenha
`.env.local` com `VITE_LIBRAS_API_URL=http://127.0.0.1:8000`.

### Limites desta primeira integracao

- As imagens precisam chegar frequentemente (ideal ~12 FPS) para que a
  reamostragem se aproxime das gravacoes de treino; lentidao da maquina
  pode dificultar o reconhecimento.
- Os dois modelos nao traduzem frases inteiras em Libras.
- A previsao temporal e experimental e depende de exemplos adequados,
  pessoas diferentes, sinais de rejeicao/nenhum sinal e validacao real.
- A API e apenas para uso local: nao exponha a internet.
- As gravacoes brutas ficam em `sequencias/`, fora do controle de versao.
