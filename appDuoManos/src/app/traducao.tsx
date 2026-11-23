import { CameraView, useCameraPermissions } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const API_URL = (process.env.EXPO_PUBLIC_API_URL || '').replace(/\/$/, '');
const CONFIANCA_MINIMA = 0.82;
const LEITURAS_ESTAVEIS = 3;
const LEITURAS_PARA_LIBERAR = 2;
const INTERVALO_MS = 650;

type Resposta = {
  sinal: string | null;
  confianca: number;
  maos_detectadas: number;
};

export default function TraducaoScreen() {
  const [permissao, solicitarPermissao] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const ativoRef = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const candidato = useRef('');
  const contador = useRef(0);
  const liberado = useRef(true);
  const semSinal = useRef(0);

  const [ativo, setAtivo] = useState(false);
  const [palavraAtual, setPalavraAtual] = useState('');
  const [confianca, setConfianca] = useState(0);
  const [frase, setFrase] = useState<string[]>([]);
  const [aviso, setAviso] = useState(
    'Conecte a API Python e toque em Iniciar reconhecimento.'
  );

  useEffect(() => {
    return () => {
      ativoRef.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  function registrarResultado(resultado: Resposta) {
    const sinal =
      resultado.sinal && resultado.confianca >= CONFIANCA_MINIMA
        ? resultado.sinal.trim()
        : '';

    setPalavraAtual(sinal);
    setConfianca(resultado.confianca);

    if (!sinal) {
      candidato.current = '';
      contador.current = 0;
      semSinal.current += 1;
      if (semSinal.current >= LEITURAS_PARA_LIBERAR) {
        liberado.current = true;
      }
      return;
    }

    semSinal.current = 0;
    if (candidato.current === sinal) {
      contador.current += 1;
    } else {
      candidato.current = sinal;
      contador.current = 1;
    }

    if (contador.current >= LEITURAS_ESTAVEIS && liberado.current) {
      setFrase((anterior) => [...anterior, sinal.toUpperCase()]);
      liberado.current = false;
    }
  }

  async function capturarEReconhecer() {
    if (!ativoRef.current) return;
    try {
      const foto = await camera.current?.takePictureAsync({
        quality: 0.35,
        base64: false,
        skipProcessing: false,
      });
      if (!foto || !ativoRef.current) return;

      const formulario = new FormData();
      formulario.append(
        'imagem',
        { uri: foto.uri, name: 'captura.jpg', type: 'image/jpeg' } as any
      );

      const resposta = await fetch(`${API_URL}/reconhecer`, {
        method: 'POST',
        body: formulario,
      });
      if (!resposta.ok) {
        throw new Error(`API retornou HTTP ${resposta.status}`);
      }
      const dados: Resposta = await resposta.json();
      if (!ativoRef.current) return;
      registrarResultado(dados);
      setAviso(dados.maos_detectadas ? 'Analisando sinais...' : 'Mostre as mãos à câmera.');
    } catch (erro) {
      if (ativoRef.current) {
        setAviso(
          `Não foi possível conectar à API. Confira o IP e a rede Wi-Fi. (${String(erro)})`
        );
      }
    } finally {
      if (ativoRef.current) {
        timer.current = setTimeout(capturarEReconhecer, INTERVALO_MS);
      }
    }
  }

  function alternarReconhecimento() {
    if (ativoRef.current) {
      ativoRef.current = false;
      setAtivo(false);
      if (timer.current) clearTimeout(timer.current);
      setAviso('Reconhecimento pausado.');
      return;
    }

    if (!API_URL) {
      setAviso('Configure EXPO_PUBLIC_API_URL no arquivo .env do aplicativo.');
      return;
    }

    candidato.current = '';
    contador.current = 0;
    semSinal.current = 0;
    liberado.current = true;
    ativoRef.current = true;
    setAtivo(true);
    setAviso('Iniciando câmera...');
    void capturarEReconhecer();
  }

  if (!permissao) {
    return <SafeAreaView style={styles.centralizado}><ActivityIndicator /></SafeAreaView>;
  }

  if (!permissao.granted) {
    return (
      <SafeAreaView style={styles.centralizado}>
        <Text style={styles.titulo}>Permissão da câmera</Text>
        <Text style={styles.info}>
          O DuoManos precisa da câmera para reconhecer os sinais.
        </Text>
        <TouchableOpacity style={styles.botao} onPress={solicitarPermissao}>
          <Text style={styles.textoBotao}>Permitir câmera</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.titulo}>Tradução ao vivo</Text>
      <Text style={styles.info}>
        Protótipo: reconhece sinais isolados cadastrados no modelo.
      </Text>

      <View style={styles.cameraContainer}>
        <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="front" />
      </View>

      <Text style={styles.aviso}>{aviso}</Text>
      <View style={styles.resultado}>
        <Text style={styles.rotulo}>Sinal detectado</Text>
        <Text style={styles.palavra}>{palavraAtual || '—'}</Text>
        <Text style={styles.info}>
          {palavraAtual ? `Confiança: ${Math.round(confianca * 100)}%` : 'Aguardando sinal'}
        </Text>
      </View>

      <Text style={styles.rotulo}>Texto reconhecido</Text>
      <ScrollView style={styles.fraseContainer}>
        <Text style={styles.frase}>{frase.join(' ') || 'As palavras aparecerão aqui.'}</Text>
      </ScrollView>

      <TouchableOpacity
        style={[styles.botao, ativo && styles.botaoParar]}
        onPress={alternarReconhecimento}
      >
        <Text style={styles.textoBotao}>
          {ativo ? 'Parar reconhecimento' : 'Iniciar reconhecimento'}
        </Text>
      </TouchableOpacity>

      <View style={styles.linha}>
        <TouchableOpacity
          style={styles.botaoSecundario}
          onPress={() => setFrase((anterior) => anterior.slice(0, -1))}
        >
          <Text>Apagar última</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.botaoSecundario}
          onPress={() => setFrase([])}
        >
          <Text>Limpar texto</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, gap: 10, backgroundColor: '#f7faff' },
  centralizado: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24 },
  titulo: { fontSize: 25, fontWeight: 'bold', color: '#173b70' },
  info: { color: '#52657a', textAlign: 'center' },
  cameraContainer: { flex: 3, minHeight: 180, overflow: 'hidden', borderRadius: 16, backgroundColor: '#121212' },
  aviso: { color: '#334155', textAlign: 'center', fontSize: 12 },
  resultado: { alignItems: 'center', backgroundColor: 'white', borderRadius: 12, padding: 10 },
  rotulo: { fontWeight: '600', color: '#475569', marginTop: 3 },
  palavra: { fontSize: 26, fontWeight: 'bold', color: '#206edf' },
  fraseContainer: { flex: 1, minHeight: 55, backgroundColor: 'white', borderRadius: 12, padding: 12 },
  frase: { fontSize: 17, color: '#0f172a' },
  botao: { padding: 13, borderRadius: 12, alignItems: 'center', backgroundColor: '#2563eb' },
  botaoParar: { backgroundColor: '#be3144' },
  textoBotao: { color: 'white', fontWeight: 'bold' },
  linha: { flexDirection: 'row', gap: 10 },
  botaoSecundario: { flex: 1, padding: 11, backgroundColor: '#e2e8f0', borderRadius: 10, alignItems: 'center' },
});
