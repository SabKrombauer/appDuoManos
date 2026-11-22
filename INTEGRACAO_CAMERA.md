# Protótipo: câmera do React Native + MediaPipe Python

A implementação está em **branch separada**; a branch `main` permanece inalterada.

## O que foi integrado
- `appDuoManos/src/app/traducao.tsx`: câmera frontal, envio periódico de fotos para API, filtro simples de previsões e histórico.
- `api.py`: recebe imagens em `POST /prever`, detecta até duas mãos e executa `modelo_libras.pkl` usando **a mesma normalização de 126 coordenadas** dos scripts originais.
- `appDuoManos/package.json` e `appDuoManos/app.json`: dependência e permissão da câmera.

## 1. Rodar a API no computador
Na raiz do repositório:

```powershell
python -m pip install fastapi uvicorn mediapipe opencv-python numpy scikit-learn
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

É necessário ter `modelo_libras.pkl` **gerado por você** e `hand_landmarker.task` ao lado de `api.py`. Se o modelo não estiver presente, execute `python treinar_modelo.py`. Se o landmarker estiver ausente, execute `python coleta_libras.py` para baixá-lo (ou baixe o modelo oficial MediaPipe Hand Landmarker).

Teste no computador: http://127.0.0.1:8000/saude

> Só carregue arquivos `.pkl` de sua confiança: o formato pickle executa código ao ser carregado.

## 2. Rodar o aplicativo
No terminal:

```powershell
cd appDuoManos
npm install
npx expo install --check
npx expo start
```

A instalação atualiza o lockfile local. Faça commit da atualização se desejar manter o travamento de dependências sincronizado.

No celular, conceda permissão à câmera, conecte-o à **mesma rede Wi-Fi** do computador e altere na tela o endereço da API para `http://IP_DO_COMPUTADOR:8000`. Veja o IP pelo comando `ipconfig` no Windows.

Se não conectar, verifique firewall, isolamento de rede Wi-Fi, se a API está rodando e se o sistema Android permite requisições HTTP locais no seu tipo de build. Use apenas em **rede privada de desenvolvimento**: a API não possui autenticação nem TLS. Não a exponha na internet.

## Limitações
- Esta versão captura imagens em intervalos (não vídeo contínuo) e só reconhece classes do modelo treinado.
- O processamento das imagens ocorre no computador; o APK ainda não é autônomo/offline.
- A câmera de webcam original usava imagem espelhada. A API espelha a foto para tentar preservar o padrão. Se as classes ficarem trocadas, teste remover `cv2.flip(frame, 1)` em `api.py`.
- A classe "desconhecido" e ausência de mãos não são adicionadas ao histórico.
- As dependências e a integração **não foram testadas em um dispositivo físico**.
- Os vídeos locais e a rota de português para Libras foram preservados, para não quebrar funcionalidades atuais.
