# Integração da câmera do DuoManos (protótipo)

Esta branch mantém o modelo Random Forest original e os scripts de coleta/treinamento.
A câmera do Expo envia **uma foto de cada vez** à API local. Não é tradução
contínua de Libras nem reconhecimento de movimentos complexos.

## 1. Inicie a API (no computador)

No diretório raiz do repositório, com Python instalado:

```bash
python -m pip install -r requirements-api.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Verifique `http://127.0.0.1:8000/saude`. Deve responder `{"status":"ok"}`.
O arquivo `modelo_libras.pkl` foi produzido por `treinar_modelo.py` e
`hand_landmarker.task` é o modelo do MediaPipe. Ambos devem estar na raiz.
Nunca carregue arquivos .pkl de fontes desconhecidas.

## 2. Instale a câmera do Expo

```bash
cd appDuoManos
npm install
npx expo install expo-camera
```

O comando `expo install` escolhe uma versão compatível com o Expo do projeto
e atualiza o package.json/lockfile local.

Copie `.env.example` para `.env` e troque `192.168.0.100` pelo endereço
IPv4 do seu computador na rede local. No Windows: `ipconfig`.

Exemplo:

```env
EXPO_PUBLIC_API_URL=http://192.168.0.105:8000
```

Reinicie o Expo após alterar a variável:

```bash
npx expo start -c
```

Abra o aplicativo no Android, conceda acesso à câmera e entre em
**Tradução ao vivo**. Use um dispositivo físico e mantenha-o na mesma Wi-Fi
do computador. Se a conexão falhar, confirme se o firewall permite conexões
locais à porta 8000.

## Como funciona

- A câmera frontal tira uma foto a cada resposta + intervalo de 650 ms.
- O Python espelha a foto para reproduzir o pré-processamento do treinamento.
- MediaPipe detecta até duas mãos e gera 126 coordenadas normalizadas.
- Random Forest retorna sinal e confiança.
- O app aceita o mesmo sinal após três leituras estáveis (>= 82%) e
  espera duas leituras sem sinal válido antes de permitir outra palavra,
  evitando repetição contínua.

As regras de estabilidade são aproximações para teste, não garantias de
correção. Elas podem precisar de ajustes na prática.

## Limitações e segurança

- Este protótipo depende do computador com Python ligado.
- Só reconhece as classes do modelo atual; **não traduz Libras completa**.
- Não há autenticação na API: utilize apenas rede privada de testes. Nunca
  exponha a porta 8000 diretamente à internet.
- Fotos são processadas em memória pela API, sem salvamento implementado.
- Para APK independente do computador, será necessário portar o modelo
  de reconhecimento para execução no dispositivo ou hospedar API segura.
- Android pode bloquear HTTP sem criptografia em builds de produção; é
  preciso configurar HTTPS e segurança de rede antes de publicar.
- Os recursos antigos de vídeos foram mantidos nesta branch, para evitar
  quebrar telas existentes; a remoção pode ser feita após validar a câmera.

## Teste rápido

1. Confirme `/saude` no navegador do computador.
2. No navegador do celular, tente `http://IP_DO_COMPUTADOR:8000/saude`.
3. Abra a tela de tradução, toque em Iniciar e faça um dos sinais já treinados.
4. Se houver baixa confiança, teste o espelhamento e iluminação.
