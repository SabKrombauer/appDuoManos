import { CameraView, useCameraPermissions } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

type Prediction = {
  sinal: string | null;
  confianca: number;
  maos_detectadas: number;
};
const INTERVAL_MS = 750;
const REQUIRED_MATCHES = 3;

export default function TraducaoScreen() {
  const camera = useRef<CameraView>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const running = useRef(false);
  const busy = useRef(false);
  const candidate = useRef('');
  const matches = useRef(0);
  const lastAccepted = useRef('');
  const [permission, requestPermission] = useCameraPermissions();
  const [ready, setReady] = useState(false);
  const [active, setActive] = useState(false);
  const [apiUrl, setApiUrl] = useState('http://192.168.0.10:8000');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [error, setError] = useState('');

  const stop = () => {
    running.current = false;
    setActive(false);
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    candidate.current = '';
    matches.current = 0;
    lastAccepted.current = '';
  };

  useEffect(() => () => {
    running.current = false;
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const analyze = async () => {
    if (!running.current || busy.current || !camera.current || !ready) return;
    busy.current = true;
    try {
      const photo = await camera.current.takePictureAsync({
        base64: true, quality: 0.35, shutterSound: false,
      });
      if (!photo?.base64) throw new Error('A câmera não retornou uma imagem.');
      const endpoint = apiUrl.trim().replace(/\/+$/, '') + '/prever';
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      let response: Response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imagem: photo.base64 }),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
      }
      if (!response.ok) throw new Error('API retornou HTTP ' + response.status);
      const data = (await response.json()) as Prediction;
      if (!running.current) return;
      setPrediction(data);
      setError('');
      const label = data.sinal && data.confianca >= 0.75 ? data.sinal : '';
      if (!label || label.toLowerCase() === 'desconhecido') {
        candidate.current = '';
        matches.current = 0;
        lastAccepted.current = '';
      } else {
        if (candidate.current === label) matches.current += 1;
        else { candidate.current = label; matches.current = 1; }
        if (matches.current >= REQUIRED_MATCHES && lastAccepted.current !== label) {
          setHistory(prev => [...prev, label]);
          lastAccepted.current = label;
        }
      }
    } catch (e) {
      if (running.current) setError(e instanceof Error ? e.message : 'Falha na API');
    } finally {
      busy.current = false;
      if (running.current) timer.current = setTimeout(analyze, INTERVAL_MS);
    }
  };

  const start = () => {
    if (!ready || active) return;
    setError('');
    candidate.current = '';
    matches.current = 0;
    lastAccepted.current = '';
    running.current = true;
    setActive(true);
    void analyze();
  };

  if (!permission) return <SafeAreaView style={styles.center}><ActivityIndicator /></SafeAreaView>;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.title}>Permissão da câmera</Text>
        <Text style={styles.help}>Precisamos da câmera para reconhecer sinais.</Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Permitir câmera</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Tradução ao vivo</Text>
      <Text style={styles.help}>Protótipo: reconhece somente os sinais treinados no seu modelo.</Text>
      <CameraView
        ref={camera} style={styles.camera} facing="front"
        onCameraReady={() => setReady(true)}
        onMountError={event => setError(event.message)}
      />
      <Text style={styles.label}>Endereço da API Python (IP do computador)</Text>
      <TextInput
        style={styles.input} value={apiUrl} onChangeText={setApiUrl}
        autoCapitalize="none" autoCorrect={false} keyboardType="url"
        editable={!active} placeholder="http://192.168.0.10:8000"
      />
      <View style={styles.row}>
        <Pressable style={[styles.button, active && styles.stop]} onPress={active ? stop : start}>
          <Text style={styles.buttonText}>{active ? 'Parar' : 'Iniciar reconhecimento'}</Text>
        </Pressable>
        <Pressable style={styles.secondary} onPress={() => setHistory(prev => prev.slice(0, -1))}>
          <Text>Apagar última</Text>
        </Pressable>
        <Pressable style={styles.secondary} onPress={() => setHistory([])}>
          <Text>Limpar</Text>
        </Pressable>
      </View>
      <Text style={styles.status}>
        {prediction?.sinal
          ? `Detectado: ${prediction.sinal} (${Math.round(prediction.confianca * 100)}%)`
          : active ? 'Aguardando sinal...' : 'Reconhecimento pausado'}
      </Text>
      {!!error && <Text style={styles.error}>{error}</Text>}
      <Text style={styles.label}>Palavras reconhecidas</Text>
      <ScrollView style={styles.result}>
        <Text style={styles.resultText}>{history.join(' ') || 'Os sinais aparecerão aqui.'}</Text>
      </ScrollView>
      <Text style={styles.help}>As imagens são enviadas pela rede local ao Python. Não use a API em redes públicas.</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 14, gap: 10, backgroundColor: '#fff' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 14 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#182238' },
  help: { color: '#596579', fontSize: 12 },
  camera: { width: '100%', flex: 1, minHeight: 180, borderRadius: 12, overflow: 'hidden' },
  label: { fontWeight: '600', fontSize: 14 },
  input: { borderColor: '#abb6c9', borderWidth: 1, borderRadius: 8, padding: 10 },
  row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 8 },
  button: { backgroundColor: '#2563eb', padding: 12, borderRadius: 9 },
  stop: { backgroundColor: '#a83232' },
  buttonText: { color: '#fff', fontWeight: 'bold' },
  secondary: { backgroundColor: '#e9edf4', borderRadius: 8, padding: 11 },
  status: { color: '#223750' },
  error: { color: '#b91c1c' },
  result: { flex: 0.35, backgroundColor: '#eff4ff', borderRadius: 10, padding: 12 },
  resultText: { fontSize: 17, color: '#13263f' },
});
