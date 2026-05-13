"""
Sistema de Monitoramento de Bem-Estar para Teletrabalho — SBRT 2026

Versão refatorada com correções:
  (a) Documentação dos pesos heurísticos
  (b) Histerese no alerta de emoções negativas
  (c) Baseline pessoal de BPM (substitui constante 75)
  (d) Filtro de qualidade do rPPG antes de usar BPM
  (e) Thread worker para DeepFace (não bloqueia loop principal)
  (f) Warmup do DeepFace
  (g) Calibração de EAR por pessoa nos primeiros segundos
  (h) Termo de consentimento LGPD na inicialização
  (i) Correção do O(n²) em list().index()
  (j) Métrica PERCLOS (percent eye closure) em janela rolante

NOTA CIENTÍFICA: O índice de "Risco de Burnout" calculado aqui é uma
heurística composta exploratória, derivada de indicadores afetivos e
fisiológicos. NÃO é um instrumento clínico validado. Validação contra o
Maslach Burnout Inventory (MBI) é necessária antes de qualquer uso fora
de pesquisa.
"""

import cv2
from deepface import DeepFace
from collections import Counter, deque
import pandas as pd
import datetime
import os
import sys
import json
import logging
import importlib
import threading
from queue import Queue, Empty
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform

try:
    mediapipe = importlib.import_module("mediapipe")
except ImportError:
    mediapipe = None

# ─── Configurações ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("monitoramento")

LOG_FILE = "monitoramento_emocional.csv"
CONFIG_FILE = "config_monitoramento.json"
CONSENTIMENTO_FLAG = "consentimento_lgpd.flag"
FRAMES_PISCAR = 3
CSV_BUFFER_SIZE = 32

DEFAULT_CONFIG = {
    "limiares": {
        "alerta_segundos": 180,
        "piscadas_min": 25,
        "burnout_baixo": 25,
        "burnout_moderado": 50,
        "burnout_alto": 75,
        "perclos_alerta": 15.0,        # % - acima disso indica fadiga (literatura: PERCLOS 15%)
        "perclos_janela_segundos": 60,
    },
    "pesos_wellness": {
        # NOTA: pesos heurísticos. Felicidade pontua positivo; emoções negativas
        # pontuam negativo, com magnitudes maiores para raiva/medo (literatura de
        # neurofisiologia do burnout aponta alteração no processamento de ameaça).
        # Surpresa é levemente positiva. Neutral leve negativo capta o efeito
        # documentado de "achatamento afetivo" em burnout (Tominaga et al., 2023).
        "happy": 1.0,
        "neutral": -0.1,
        "surprise": 0.2,
        "sad": -0.6,
        "fear": -0.7,
        "disgust": -0.5,
        "angry": -0.8,
    },
    "detector_rosto": {
        "modelo": "yolov8n-face.pt",
        "confianca": 0.35,
        "imgsz": 640,
    },
    "rppg": {
        "janela_segundos": 12,
        "amostras_minimas": 30,
        "bpm_min": 45,
        "bpm_max": 180,
        "qualidade_minima": 0.04,   # razão da potência espectral concentrada no pico
    },
    "calibracao": {
        "ear_segundos": 10,            # tempo para calibrar EAR baseline pessoal
        "ear_amostras_minimas": 30,
        "bpm_atraso_segundos": 20,     # espera estabilizar antes de coletar baseline
        "bpm_segundos": 30,            # duração da coleta do baseline BPM
        "bpm_default": 75.0,           # fallback se calibração falhar
    },
    "lgpd": {
        "exigir_consentimento": True,
    },
}


def carregar_configuracoes():
    """Carrega configuração externa e aplica fallback seguro para defaults."""
    if not os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
            logger.info(f"Arquivo de configuração criado: {CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Falha ao criar {CONFIG_FILE}: {e}")
        return {k: dict(v) for k, v in DEFAULT_CONFIG.items()}

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)

        merged = {}
        for chave, padrao in DEFAULT_CONFIG.items():
            merged[chave] = {**padrao, **user_cfg.get(chave, {})}
        return merged
    except Exception as e:
        logger.warning(f"Falha ao ler {CONFIG_FILE}; usando defaults. Motivo: {e}")
        return {k: dict(v) for k, v in DEFAULT_CONFIG.items()}


CONFIG = carregar_configuracoes()

LIMITE_ALERTA_SEGUNDOS = int(CONFIG["limiares"]["alerta_segundos"])
LIMIAR_PISCADAS_MIN = int(CONFIG["limiares"]["piscadas_min"])
LIMIAR_BURNOUT_BAIXO = float(CONFIG["limiares"]["burnout_baixo"])
LIMIAR_BURNOUT_MODERADO = float(CONFIG["limiares"]["burnout_moderado"])
LIMIAR_BURNOUT_ALTO = float(CONFIG["limiares"]["burnout_alto"])
LIMIAR_PERCLOS_ALERTA = float(CONFIG["limiares"]["perclos_alerta"])
PERCLOS_JANELA_SEGUNDOS = int(CONFIG["limiares"]["perclos_janela_segundos"])
YOLO_FACE_MODELO = str(CONFIG["detector_rosto"]["modelo"])
YOLO_FACE_CONF = float(CONFIG["detector_rosto"]["confianca"])
YOLO_FACE_IMGSZ = int(CONFIG["detector_rosto"]["imgsz"])
RPPG_JANELA_SEGUNDOS = int(CONFIG["rppg"]["janela_segundos"])
RPPG_AMOSTRAS_MINIMAS = int(CONFIG["rppg"]["amostras_minimas"])
RPPG_BPM_MIN = int(CONFIG["rppg"]["bpm_min"])
RPPG_BPM_MAX = int(CONFIG["rppg"]["bpm_max"])
RPPG_QUALIDADE_MIN = float(CONFIG["rppg"]["qualidade_minima"])
CAL_EAR_SEGUNDOS = int(CONFIG["calibracao"]["ear_segundos"])
CAL_EAR_AMOSTRAS_MIN = int(CONFIG["calibracao"]["ear_amostras_minimas"])
CAL_BPM_ATRASO = int(CONFIG["calibracao"]["bpm_atraso_segundos"])
CAL_BPM_SEGUNDOS = int(CONFIG["calibracao"]["bpm_segundos"])
CAL_BPM_DEFAULT = float(CONFIG["calibracao"]["bpm_default"])
EXIGIR_CONSENTIMENTO = bool(CONFIG["lgpd"]["exigir_consentimento"])

EMOCAO_MAP = {
    'happy':    ('Feliz',    (0, 220, 0)),
    'neutral':  ('Neutro',   (200, 200, 200)),
    'sad':      ('Triste',   (220, 80, 0)),
    'angry':    ('Raiva',    (0, 0, 220)),
    'fear':     ('Medo',     (0, 140, 255)),
    'disgust':  ('Desgosto', (0, 100, 100)),
    'surprise': ('Surpresa', (255, 220, 0)),
}

EMOCOES_NEGATIVAS = {'sad', 'angry', 'fear', 'disgust'}

PESOS_WELLNESS = {k: float(v) for k, v in CONFIG["pesos_wellness"].items()}

LEFT_EYE_IDXS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDXS = (362, 385, 387, 263, 373, 380)

EAR_THRESHOLD_DEFAULT = 0.20  # fallback antes da calibração

# ─── (h) LGPD — Termo de consentimento ──────────────────────────────────────────

TERMO_CONSENTIMENTO = """
═══════════════════════════════════════════════════════════════════════
  TERMO DE CONSENTIMENTO — LGPD (Lei nº 13.709/2018)
═══════════════════════════════════════════════════════════════════════

Este sistema realiza coleta CONTÍNUA de dados biométricos sensíveis:
  • Imagens faciais (não armazenadas, processadas em memória)
  • Expressões emocionais inferidas (DeepFace + FER2013)
  • Taxa de piscadas e Eye Aspect Ratio (EAR)
  • Frequência cardíaca estimada por rPPG
  • PERCLOS (percentual de fechamento ocular)

Estes dados constituem categoria especial (Art. 11 da LGPD) e exigem
consentimento livre, informado e inequívoco. Os dados agregados são
salvos em CSV local; imagens NÃO são gravadas.

FINALIDADE: pesquisa exploratória sobre indicadores de bem-estar em
teletrabalho. O índice de "Risco de Burnout" é HEURÍSTICO e não tem
valor diagnóstico clínico.

DIREITOS DO TITULAR (Arts. 17-22): acesso, correção, anonimização,
portabilidade, eliminação e revogação do consentimento a qualquer
momento, deletando o arquivo "{flag}" e os CSVs gerados.

═══════════════════════════════════════════════════════════════════════
""".format(flag=CONSENTIMENTO_FLAG)


def verificar_consentimento():
    """Verifica/solicita consentimento LGPD antes de iniciar a coleta."""
    if not EXIGIR_CONSENTIMENTO:
        return True
    if os.path.isfile(CONSENTIMENTO_FLAG):
        return True
    print(TERMO_CONSENTIMENTO)
    try:
        resp = input("Aceita os termos acima? [s/N]: ").strip().lower()
    except EOFError:
        resp = ''
    if resp != 's':
        print("Consentimento NÃO concedido. Encerrando.")
        return False
    try:
        with open(CONSENTIMENTO_FLAG, 'w', encoding='utf-8') as f:
            f.write(f"Consentimento concedido em {datetime.datetime.now().isoformat()}\n")
    except Exception as e:
        logger.warning(f"Falha ao registrar consentimento: {e}")
    return True


# ─── Detectores ─────────────────────────────────────────────────────────────────

def carregar_face_mesh():
    if mediapipe is None:
        logger.warning("MediaPipe não está instalado; usando fallback de olhos com Haar.")
        return None
    try:
        return mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        logger.warning(f"Falha ao inicializar MediaPipe Face Mesh: {e}")
        return None


def carregar_detector_rosto_yolo():
    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca ultralytics não está instalada. "
            "Instale com 'pip install ultralytics' para usar o detector YOLO."
        ) from e
    try:
        return ultralytics.YOLO(YOLO_FACE_MODELO)
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível carregar o modelo YOLO '{YOLO_FACE_MODELO}': {e}"
        ) from e


def detectar_faces_yolo(frame, detector):
    try:
        resultados = detector.predict(
            source=frame,
            conf=YOLO_FACE_CONF,
            imgsz=YOLO_FACE_IMGSZ,
            verbose=False,
        )
        faces = []
        for resultado in resultados:
            boxes = getattr(resultado, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                if conf < YOLO_FACE_CONF:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(round(x1)))
                y1 = max(0, int(round(y1)))
                x2 = min(frame.shape[1], int(round(x2)))
                y2 = min(frame.shape[0], int(round(y2)))
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    faces.append((x1, y1, w, h))
        faces.sort(key=lambda item: (item[0], item[1]))
        return faces
    except Exception as e:
        if not getattr(detectar_faces_yolo, '_erro_reportado', False):
            logger.error(f"Falha na detecção YOLO de rostos: {e}")
            detectar_faces_yolo._erro_reportado = True
        return []


# ─── EAR e olhos ────────────────────────────────────────────────────────────────

def _ponto_landmark(landmark, idx, largura, altura):
    ponto = landmark[idx]
    return np.array([ponto.x * largura, ponto.y * altura], dtype=np.float32)


def _distancia_pontos(a, b):
    return float(np.linalg.norm(a - b))


def calcular_ear(landmark, largura, altura):
    olho_esquerdo = [_ponto_landmark(landmark, idx, largura, altura) for idx in LEFT_EYE_IDXS]
    olho_direito = [_ponto_landmark(landmark, idx, largura, altura) for idx in RIGHT_EYE_IDXS]

    def ear(olho):
        v1 = _distancia_pontos(olho[1], olho[5])
        v2 = _distancia_pontos(olho[2], olho[4])
        h = _distancia_pontos(olho[0], olho[3])
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    return (ear(olho_esquerdo) + ear(olho_direito)) / 2.0


def detectar_olhos(face_roi, face_mesh_inst=None, ear_threshold=EAR_THRESHOLD_DEFAULT):
    """Retorna (olhos_abertos, ear, landmarks). EAR pode ser None se fallback."""
    if face_roi.size == 0:
        return False, None, None

    if face_mesh_inst is not None:
        try:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resultado = face_mesh_inst.process(rgb)
            if resultado.multi_face_landmarks:
                landmarks = resultado.multi_face_landmarks[0].landmark
                ear = calcular_ear(landmarks, face_roi.shape[1], face_roi.shape[0])
                return ear > ear_threshold, ear, landmarks
        except Exception as e:
            if not getattr(detectar_olhos, '_erro_reportado', False):
                logger.warning(f"Falha no MediaPipe Face Mesh para olhos ({e}).")
                detectar_olhos._erro_reportado = True

    # Fallback Haar
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        regiao_olhos = gray[:int(h * 0.6), :]
        olhos = eye_cascade_fallback.detectMultiScale(regiao_olhos, 1.1, 3, minSize=(15, 15))
        return len(olhos) > 0, None, None
    except Exception as e:
        if not getattr(detectar_olhos, '_fallback_reportado', False):
            logger.warning(f"Falha no fallback de olhos ({e}).")
            detectar_olhos._fallback_reportado = True
        return False, None, None


# ─── rPPG ───────────────────────────────────────────────────────────────────────

def extrair_amostra_rppg(face_roi, landmark=None):
    if face_roi.size == 0:
        return None
    altura, largura = face_roi.shape[:2]
    if landmark is not None:
        xs = np.array([p.x * largura for p in landmark], dtype=np.float32)
        ys = np.array([p.y * altura for p in landmark], dtype=np.float32)
        x_min = int(np.clip(xs.min(), 0, largura - 1))
        x_max = int(np.clip(xs.max(), 1, largura))
        y_min = int(np.clip(ys.min(), 0, altura - 1))
        y_max = int(np.clip(ys.max(), 1, altura))
    else:
        x_min, y_min, x_max, y_max = 0, 0, largura, altura

    box_largura = max(1, x_max - x_min)
    box_altura = max(1, y_max - y_min)

    def recorte(x1r, y1r, x2r, y2r):
        x1 = int(np.clip(x_min + box_largura * x1r, 0, largura - 1))
        y1 = int(np.clip(y_min + box_altura * y1r, 0, altura - 1))
        x2 = int(np.clip(x_min + box_largura * x2r, x1 + 1, largura))
        y2 = int(np.clip(y_min + box_altura * y2r, y1 + 1, altura))
        return face_roi[y1:y2, x1:x2]

    regioes = [
        recorte(0.25, 0.08, 0.75, 0.26),
        recorte(0.08, 0.30, 0.32, 0.62),
        recorte(0.68, 0.30, 0.92, 0.62),
    ]
    amostras = [r.mean(axis=(0, 1)) for r in regioes if r.size > 0]
    if not amostras:
        return None
    return np.mean(amostras, axis=0)


def estimar_frequencia_cardiaca(amostras):
    if len(amostras) < RPPG_AMOSTRAS_MINIMAS:
        return None, None
    tempos = np.array([item[0] for item in amostras], dtype=np.float64)
    sinais = np.array([item[1] for item in amostras], dtype=np.float64)
    duracao = tempos[-1] - tempos[0]
    if duracao < RPPG_JANELA_SEGUNDOS or duracao <= 0:
        return None, None
    fs_estimado = (len(tempos) - 1) / duracao
    if fs_estimado < 4.0:
        return None, None
    tempos_uniformes = np.linspace(tempos[0], tempos[-1], max(len(tempos), int(duracao * 30)))
    sinais_uniformes = np.interp(tempos_uniformes, tempos, sinais)
    sinais_uniformes = sinais_uniformes - np.mean(sinais_uniformes)
    if np.allclose(sinais_uniformes, 0):
        return None, None
    janela = np.hanning(len(sinais_uniformes))
    espectro = np.fft.rfft(sinais_uniformes * janela)
    frequencias = np.fft.rfftfreq(len(sinais_uniformes), d=(tempos_uniformes[1] - tempos_uniformes[0]))
    banda = (frequencias >= (RPPG_BPM_MIN / 60.0)) & (frequencias <= (RPPG_BPM_MAX / 60.0))
    if not np.any(banda):
        return None, None
    potencias = np.abs(espectro[banda]) ** 2
    if potencias.size == 0:
        return None, None
    indice = int(np.argmax(potencias))
    bpm = float(frequencias[banda][indice] * 60.0)
    qualidade = float(potencias[indice] / (potencias.sum() + 1e-9))
    if not (RPPG_BPM_MIN <= bpm <= RPPG_BPM_MAX):
        return None, None
    return round(bpm, 1), round(qualidade, 3)


# ─── (e) Worker thread para DeepFace ────────────────────────────────────────────

class EmotionWorker:
    """Executa DeepFace em thread separada para não bloquear o loop principal.

    Aceita no máximo uma análise por pid em vôo. Resultados são drenados
    pelo loop principal a cada iteração via get_results().
    """

    def __init__(self):
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.pids_pendentes = set()
        self.lock = threading.Lock()

    def start(self):
        self.thread.start()

    def submit(self, pid, face_roi):
        """Submete um rosto para análise. Retorna False se já há análise em vôo para esse pid."""
        if face_roi is None or face_roi.size == 0:
            return False
        with self.lock:
            if pid in self.pids_pendentes:
                return False
            self.pids_pendentes.add(pid)
        # Copia para evitar mutação do frame enquanto a thread processa
        self.in_queue.put((pid, face_roi.copy()))
        return True

    def get_results(self):
        """Drena resultados disponíveis. Retorna lista de (pid, resultado, erro)."""
        resultados = []
        while True:
            try:
                pid, res, err = self.out_queue.get_nowait()
            except Empty:
                break
            with self.lock:
                self.pids_pendentes.discard(pid)
            resultados.append((pid, res, err))
        return resultados

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                pid, face_roi = self.in_queue.get(timeout=0.3)
            except Empty:
                continue
            try:
                res = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                )
                self.out_queue.put((pid, res, None))
            except Exception as e:
                self.out_queue.put((pid, None, str(e)))


# ─── CSV ────────────────────────────────────────────────────────────────────────

COLUNAS_CSV = [
    'Timestamp', 'PessoaID', 'Emotion', 'Confidence',
    'Wellness', 'BurnoutRisk', 'HeartRateBPM', 'HeartRateQuality',
    'PERCLOS', 'EAR'
]


def inicializar_csv():
    if os.path.isfile(LOG_FILE):
        try:
            cabecalho = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
            if cabecalho != COLUNAS_CSV:
                backup = LOG_FILE.replace('.csv', f"_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(LOG_FILE, backup)
                logger.warning(f"Formato antigo salvo em: {backup}")
                pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)
        except Exception as e:
            logger.error(f"Erro ao validar CSV: {e}")
            pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)
    else:
        pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)


def salvar_registros_csv(registros):
    if not registros:
        return
    try:
        pd.DataFrame(registros, columns=COLUNAS_CSV).to_csv(
            LOG_FILE, mode='a', header=False, index=False
        )
    except Exception as e:
        logger.error(f"Erro ao salvar CSV: {e}")


# ─── Cálculos ───────────────────────────────────────────────────────────────────

def tocar_alerta():
    sistema = platform.system()
    if sistema == 'Darwin':
        os.system("afplay /System/Library/Sounds/Funk.aiff &")
    elif sistema == 'Windows':
        try:
            import winsound
            winsound.Beep(880, 600)
        except Exception:
            pass
    elif sistema == 'Linux':
        for cmd in (
            "paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga >/dev/null 2>&1 &",
            "aplay /usr/share/sounds/alsa/Front_Center.wav >/dev/null 2>&1 &",
        ):
            if os.system(cmd) == 0:
                break


def calcular_wellness(counter):
    """Score de bem-estar (0-100) derivado de pesos por emoção.

    HEURÍSTICA: combinação linear normalizada. Não validada clinicamente.
    """
    total = sum(counter.values())
    if total == 0:
        return 50.0
    soma = sum(PESOS_WELLNESS.get(e, 0) * c for e, c in counter.items())
    return round(max(0.0, min(100.0, 50.0 + (soma / total) * 50.0)), 1)


def calcular_risco_burnout(counter, alertas_fadiga=0, heart_rate_bpm=None,
                            hr_qualidade=None, bpm_baseline=None, perclos=None):
    """Índice composto exploratório de risco de burnout (0-100).

    Componentes:
      • 50 pts: proporção de emoções negativas
      • 20 pts: wellness invertido
      • 10 pts: alertas de fadiga (piscadas elevadas) acumulados
      • 10 pts: PERCLOS (literatura: >15% indica fadiga)
      • 10 pts: desvio do BPM em relação ao baseline pessoal (apenas se
                qualidade rPPG ≥ RPPG_QUALIDADE_MIN)

    HEURÍSTICA, NÃO DIAGNÓSTICO. Precisa validação contra MBI.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0

    neg = sum(counter.get(e, 0) for e in EMOCOES_NEGATIVAS)
    pontos_emocao   = (neg / total) * 50.0
    pontos_wellness = max(0.0, (100.0 - calcular_wellness(counter)) / 100.0 * 20.0)
    pontos_fadiga   = min(alertas_fadiga * 5.0, 10.0)

    # (j) PERCLOS — literatura aponta 15% como limiar de fadiga
    pontos_perclos = 0.0
    if perclos is not None:
        # Linear de 5% até 25% mapeia para 0-10 pontos
        pontos_perclos = max(0.0, min(10.0, (float(perclos) - 5.0) / 20.0 * 10.0))

    # (c)+(d) BPM com filtro de qualidade e baseline pessoal
    pontos_pulso = 0.0
    if (heart_rate_bpm is not None
            and (hr_qualidade is None or hr_qualidade >= RPPG_QUALIDADE_MIN)):
        baseline = bpm_baseline if bpm_baseline is not None else CAL_BPM_DEFAULT
        desvio = abs(float(heart_rate_bpm) - baseline)
        # Desvio de 30 bpm = 10 pts
        pontos_pulso = max(0.0, min(10.0, desvio / 30.0 * 10.0))

    return round(
        min(100.0, pontos_emocao + pontos_wellness + pontos_fadiga + pontos_perclos + pontos_pulso),
        1,
    )


def classificar_burnout(risco):
    if risco < LIMIAR_BURNOUT_BAIXO:    return "Baixo",    (0, 200, 0)
    if risco < LIMIAR_BURNOUT_MODERADO: return "Moderado", (0, 165, 255)
    if risco < LIMIAR_BURNOUT_ALTO:     return "Alto",     (30, 100, 255)
    return "Crítico", (0, 0, 220)


# ─── (b) (g) (j) Estado por pessoa ──────────────────────────────────────────────

class EstadoPessoa:
    def __init__(self, pid):
        self.pid = pid

        # Emoções
        self.emotion_buffer = deque(maxlen=5)
        self.emotion_counter = Counter()
        self.last_label = "Analisando..."
        self.last_color = (200, 200, 200)
        self.last_confidence = None

        # (b) Histerese — buffer de últimas N emoções suavizadas para decidir alerta
        self.negativa_buffer = deque(maxlen=5)
        self.inicio_negativa = None
        self.alerta_ativo = False

        # rPPG
        self.rppg_amostras = deque(maxlen=max(RPPG_JANELA_SEGUNDOS * 30, 120))
        self.heart_rate_bpm = None
        self.heart_rate_qualidade = None
        self.heart_rate_hist = deque(maxlen=5)

        # (c) Baseline BPM pessoal
        self.bpm_baseline = None
        self.bpm_amostras_calibracao = []
        self.bpm_inicio_calibracao = None

        # Piscadas e fadiga
        self.frames_sem_olho = 0
        self.piscadas = 0
        self.piscadas_janela = 0
        self.inicio_janela = datetime.datetime.now()
        self.alerta_fadiga = False
        self.alerta_fadiga_timestamp = None
        self.alertas_fadiga_total = 0

        # (g) Calibração EAR
        self.ear_threshold = EAR_THRESHOLD_DEFAULT
        self.ear_amostras_cal = []
        self.ear_calibracao_inicio = None
        self.ear_calibrado = False

        # (j) PERCLOS — histórico (timestamp, olho_fechado_bool) com janela rolante
        self.perclos_hist = deque()
        self.perclos_atual = 0.0

    # ─── Calibração EAR ───
    def atualizar_calibracao_ear(self, ear):
        if self.ear_calibrado or ear is None:
            return
        agora = datetime.datetime.now()
        if self.ear_calibracao_inicio is None:
            self.ear_calibracao_inicio = agora
        elapsed = (agora - self.ear_calibracao_inicio).total_seconds()
        if elapsed < CAL_EAR_SEGUNDOS:
            self.ear_amostras_cal.append(ear)
        else:
            if len(self.ear_amostras_cal) >= CAL_EAR_AMOSTRAS_MIN:
                # Percentil 75 ignora piscadas (valores baixos transientes)
                baseline = float(np.percentile(self.ear_amostras_cal, 75))
                self.ear_threshold = max(0.10, min(0.25, baseline * 0.75))
                logger.info(
                    f"P{self.pid}: EAR calibrado. Baseline={baseline:.3f}, "
                    f"threshold={self.ear_threshold:.3f}"
                )
            else:
                logger.info(f"P{self.pid}: calibração EAR insuficiente; mantendo default.")
            self.ear_calibrado = True

    # ─── Calibração BPM ───
    def atualizar_calibracao_bpm(self, bpm, qualidade):
        if self.bpm_baseline is not None or bpm is None:
            return
        if qualidade is None or qualidade < RPPG_QUALIDADE_MIN:
            return
        agora = datetime.datetime.now()
        if self.bpm_inicio_calibracao is None:
            self.bpm_inicio_calibracao = agora
            return
        elapsed = (agora - self.bpm_inicio_calibracao).total_seconds()
        if elapsed < CAL_BPM_ATRASO:
            return  # ainda esperando estabilizar
        if elapsed < CAL_BPM_ATRASO + CAL_BPM_SEGUNDOS:
            self.bpm_amostras_calibracao.append(bpm)
        else:
            if len(self.bpm_amostras_calibracao) >= 5:
                self.bpm_baseline = float(np.median(self.bpm_amostras_calibracao))
                logger.info(f"P{self.pid}: BPM baseline = {self.bpm_baseline:.1f}")
            else:
                self.bpm_baseline = CAL_BPM_DEFAULT
                logger.info(
                    f"P{self.pid}: calibração BPM falhou; usando default {CAL_BPM_DEFAULT}"
                )

    # ─── Piscadas + PERCLOS ───
    def atualizar_piscada_e_perclos(self, olho_detectado, timestamp_segundos):
        # Piscadas
        if not olho_detectado:
            self.frames_sem_olho += 1
        else:
            if self.frames_sem_olho >= FRAMES_PISCAR:
                self.piscadas += 1
                self.piscadas_janela += 1
            self.frames_sem_olho = 0

        # (j) PERCLOS rolante
        self.perclos_hist.append((timestamp_segundos, not olho_detectado))
        cutoff = timestamp_segundos - PERCLOS_JANELA_SEGUNDOS
        while self.perclos_hist and self.perclos_hist[0][0] < cutoff:
            self.perclos_hist.popleft()
        if self.perclos_hist:
            fechados = sum(1 for _, fc in self.perclos_hist if fc)
            self.perclos_atual = (fechados / len(self.perclos_hist)) * 100.0
        else:
            self.perclos_atual = 0.0

        # Janela de piscadas/minuto e alerta de fadiga
        agora = datetime.datetime.now()
        if (agora - self.inicio_janela).total_seconds() >= 60:
            if self.piscadas_janela > LIMIAR_PISCADAS_MIN:
                self.alerta_fadiga = True
                self.alerta_fadiga_timestamp = agora
                self.alertas_fadiga_total += 1
            self.piscadas_janela = 0
            self.inicio_janela = agora

        if (self.alerta_fadiga and self.alerta_fadiga_timestamp
                and (agora - self.alerta_fadiga_timestamp).total_seconds() >= 30):
            self.alerta_fadiga = False

    # ─── rPPG ───
    def atualizar_rppg(self, timestamp_segundos, amostra_rgb):
        if amostra_rgb is None:
            return self.heart_rate_bpm, self.heart_rate_qualidade
        sinal = float(amostra_rgb[1])
        self.rppg_amostras.append((timestamp_segundos, sinal))
        bpm, qualidade = estimar_frequencia_cardiaca(self.rppg_amostras)
        if bpm is not None and (qualidade is None or qualidade >= RPPG_QUALIDADE_MIN):
            self.heart_rate_hist.append(bpm)
            self.heart_rate_bpm = round(float(np.median(self.heart_rate_hist)), 1)
            self.heart_rate_qualidade = qualidade
            # (c) alimenta baseline pessoal
            self.atualizar_calibracao_bpm(self.heart_rate_bpm, qualidade)
        return self.heart_rate_bpm, self.heart_rate_qualidade

    # ─── (b) Histerese da emoção negativa ───
    def avaliar_alerta_negativo(self, emo_suavizado):
        """Retorna True se um novo alerta acaba de disparar."""
        self.negativa_buffer.append(emo_suavizado in EMOCOES_NEGATIVAS)
        if len(self.negativa_buffer) < 3:
            return False
        negativos = sum(self.negativa_buffer)
        agora = datetime.datetime.now()
        disparou = False

        # Maioria negativa (3 ou mais de 5): inicia/mantém timer
        if negativos >= 3:
            if self.inicio_negativa is None:
                self.inicio_negativa = agora
            elif (agora - self.inicio_negativa).total_seconds() >= LIMITE_ALERTA_SEGUNDOS:
                if not self.alerta_ativo:
                    self.alerta_ativo = True
                    disparou = True

        # Maioria não-negativa (≤1 de 5): zera, mas só com histerese clara
        elif negativos <= 1:
            self.inicio_negativa = None
            self.alerta_ativo = False
        # Estado ambíguo (2 de 5): mantém o que estiver
        return disparou


# ─── Tracker multi-rosto ────────────────────────────────────────────────────────

class TrackerRosto:
    def __init__(self):
        self.proxima_id = 0
        self.ids_centros = {}
        self.estados = {}
        self.frame_visto = {}
        self.historico_emocoes = Counter()
        self.historico_piscadas = 0
        self.historico_alertas_fadiga = 0
        self.historico_hr = []

    def atualizar(self, centros, frame_count, max_ausente=90):
        for rid in list(self.ids_centros):
            if frame_count - self.frame_visto.get(rid, 0) > max_ausente:
                estado_antigo = self.estados.pop(rid, None)
                if estado_antigo is not None:
                    self.historico_emocoes += estado_antigo.emotion_counter
                    self.historico_piscadas += estado_antigo.piscadas
                    self.historico_alertas_fadiga += estado_antigo.alertas_fadiga_total
                    self.historico_hr.extend(estado_antigo.heart_rate_hist)
                del self.ids_centros[rid]
                self.frame_visto.pop(rid, None)

        ids_resultado, usados = [], set()
        for centro in centros:
            melhor_id, melhor_dist = None, 220
            for rid, c_ant in self.ids_centros.items():
                if rid in usados:
                    continue
                dist = np.linalg.norm(np.array(centro) - np.array(c_ant))
                if dist < melhor_dist:
                    melhor_dist, melhor_id = dist, rid
            if melhor_id is not None:
                self.ids_centros[melhor_id] = centro
                self.frame_visto[melhor_id] = frame_count
                usados.add(melhor_id)
                ids_resultado.append(melhor_id)
            else:
                nid = self.proxima_id
                self.proxima_id += 1
                self.ids_centros[nid] = centro
                self.frame_visto[nid] = frame_count
                self.estados[nid] = EstadoPessoa(nid)
                ids_resultado.append(nid)
        return ids_resultado

    def emotion_counter_global(self):
        total = Counter(self.historico_emocoes)
        for e in self.estados.values():
            total += e.emotion_counter
        return total

    def piscadas_total(self):
        return self.historico_piscadas + sum(e.piscadas for e in self.estados.values())

    def alertas_fadiga_total(self):
        return self.historico_alertas_fadiga + sum(e.alertas_fadiga_total for e in self.estados.values())

    def heart_rate_medio(self):
        amostras = list(self.historico_hr)
        for estado in self.estados.values():
            amostras.extend(list(estado.heart_rate_hist))
        if not amostras:
            return None
        return round(float(np.median(amostras)), 1)

    def perclos_medio(self):
        valores = [e.perclos_atual for e in self.estados.values() if e.perclos_atual > 0]
        if not valores:
            return None
        return round(float(np.mean(valores)), 1)

    def bpm_baseline_medio(self):
        baselines = [e.bpm_baseline for e in self.estados.values() if e.bpm_baseline is not None]
        if not baselines:
            return None
        return float(np.median(baselines))


# ─── Relatório ──────────────────────────────────────────────────────────────────

def gerar_relatorio(emotion_counter, wellness, piscadas_total, burnout_risk=0.0,
                    heart_rate_bpm=None, perclos_medio=None):
    total = sum(emotion_counter.values())
    nivel_b = ("Baixo" if burnout_risk < LIMIAR_BURNOUT_BAIXO else
               "Moderado" if burnout_risk < LIMIAR_BURNOUT_MODERADO else
               "Alto" if burnout_risk < LIMIAR_BURNOUT_ALTO else "Crítico")
    print("\n" + "=" * 47)
    print("   RELATÓRIO DE SESSÃO — TELETRABALHO")
    print("=" * 47)
    for e, c in emotion_counter.most_common():
        label, _ = EMOCAO_MAP.get(e, (e.capitalize(), None))
        pct = c / total * 100 if total > 0 else 0
        print(f"   {label:<15}: {c:<5} ({pct:.1f}%)")
    print(f"\n   Wellness Score  : {wellness:.1f}/100")
    print(f"   Risco de Burnout: {nivel_b} ({burnout_risk:.1f}/100)")
    print(f"   Piscadas totais : {piscadas_total}")
    if heart_rate_bpm is not None:
        print(f"   FC rPPG (mediana): {heart_rate_bpm:.1f} bpm")
    if perclos_medio is not None:
        print(f"   PERCLOS médio   : {perclos_medio:.1f}%"
              f" {'(ALERTA)' if perclos_medio > LIMIAR_PERCLOS_ALERTA else ''}")
    print("=" * 47)
    print("   AVISO: índices heurísticos. Não usar como diagnóstico.")
    print(f"   Dados salvos em : {LOG_FILE}")

    if total == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Relatório Emocional — Sessão de Teletrabalho", fontsize=14, fontweight='bold')

    labels = [EMOCAO_MAP.get(e, (e.capitalize(),))[0] for e in emotion_counter]
    values = list(emotion_counter.values())
    cores_hex = ['#00DC00', '#C8C8C8', '#DC5000', '#0000DC', '#008CFF', '#006464', '#FFD600']
    axes[0].pie(values, labels=labels, autopct='%1.1f%%',
                colors=cores_hex[:len(values)], startangle=140)
    axes[0].set_title("Distribuição de Emoções")

    cor_w = '#00b300' if wellness >= 70 else '#ff9900' if wellness >= 40 else '#cc0000'
    axes[1].barh(['Bem-estar'], [wellness], color=cor_w, height=0.4)
    axes[1].barh(['Bem-estar'], [100 - wellness], left=[wellness], color='#e0e0e0', height=0.4)
    axes[1].set_xlim(0, 100)
    axes[1].set_title(f"Wellness Score: {wellness:.1f} / 100")
    axes[1].axvline(x=70, color='#00b300', linestyle='--', alpha=0.5, label='Meta (70)')
    axes[1].legend(loc='lower right')
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    cor_b = '#00b300' if burnout_risk < 25 else '#ff9900' if burnout_risk < 50 else '#ff4400' if burnout_risk < 75 else '#cc0000'
    axes[2].barh(['Burnout'], [burnout_risk], color=cor_b, height=0.4)
    axes[2].barh(['Burnout'], [100 - burnout_risk], left=[burnout_risk], color='#e0e0e0', height=0.4)
    axes[2].set_xlim(0, 100)
    axes[2].set_title(f"Risco de Burnout: {nivel_b} ({burnout_risk:.1f} / 100)")
    axes[2].axvline(x=LIMIAR_BURNOUT_ALTO, color='#cc0000', linestyle='--', alpha=0.5, label='Crítico (75)')
    axes[2].legend(loc='lower right')
    for spine in axes[2].spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    nome = f"relatorio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(nome, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Gráfico salvo   : {nome}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    global eye_cascade_fallback

    # (h) LGPD
    if not verificar_consentimento():
        sys.exit(0)

    inicializar_csv()

    # Inicializa câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Câmera não encontrada. Verifique a conexão.")
        sys.exit(1)

    # Carrega detectores
    face_detector = carregar_detector_rosto_yolo()
    face_mesh = carregar_face_mesh()
    eye_cascade_fallback = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    )

    # (f) Warmup DeepFace — primeira inferência baixa pesos e pode demorar 2-5s
    logger.info("Aquecendo DeepFace (download/carga inicial de pesos)...")
    try:
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=['emotion'],
                         enforce_detection=False, detector_backend='skip')
        logger.info("DeepFace pronto.")
    except Exception as e:
        logger.warning(f"Warmup do DeepFace falhou (continua mesmo assim): {e}")

    # (e) Worker thread
    emotion_worker = EmotionWorker()
    emotion_worker.start()

    tracker = TrackerRosto()
    frame_count = 0
    modo_privacidade = False
    buffer_csv = []
    erros_deepface = 0

    print("Iniciando Monitoramento...  |  'q' = sair  |  'p' = modo privacidade")
    print(f"Calibração ativa: EAR ({CAL_EAR_SEGUNDOS}s) + BPM "
          f"({CAL_BPM_ATRASO}+{CAL_BPM_SEGUNDOS}s)")

    # Cache do último face_roi por pid (para correlacionar com resultados assíncronos)
    ultimo_face_roi = {}
    # Cache de coords (x,y,w,h) por pid no frame atual
    coords_pid = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERRO: Não foi possível ler frame da câmera.")
                break

            frame_count += 1
            h_frame, w_frame = frame.shape[:2]
            timestamp_segundos = datetime.datetime.now().timestamp()

            faces = detectar_faces_yolo(frame, face_detector)
            centros = [((x + w // 2), (y + h // 2)) for (x, y, w, h) in faces]
            ids = tracker.atualizar(centros, frame_count)

            coords_pid.clear()
            for pid, (x, y, w, h) in zip(ids, faces):
                coords_pid[pid] = (x, y, w, h)

            # Processamento por face (rápido — sem DeepFace)
            # (i) usar enumerate em vez de list().index() para evitar O(n²)
            for indice_pid, (pid, (x, y, w, h)) in enumerate(zip(ids, faces)):
                estado = tracker.estados[pid]
                face_roi = frame[y:y+h, x:x+w]

                # Olhos + EAR (com threshold calibrado por pessoa)
                olho_detectado, ear, landmarks = (
                    detectar_olhos(face_roi, face_mesh, estado.ear_threshold)
                    if face_roi.size > 0 else (True, None, None)
                )

                # (g) Acumula amostras para calibração de EAR
                if ear is not None:
                    estado.atualizar_calibracao_ear(ear)

                # Piscadas + (j) PERCLOS
                estado.atualizar_piscada_e_perclos(olho_detectado, timestamp_segundos)

                # rPPG
                amostra_rppg = extrair_amostra_rppg(face_roi, landmarks)
                estado.atualizar_rppg(timestamp_segundos, amostra_rppg)

                # (e) Submete DeepFace a cada 8 frames (assíncrono)
                if frame_count % 8 == 0 and face_roi.size > 0:
                    if emotion_worker.submit(pid, face_roi):
                        ultimo_face_roi[pid] = (x, y, w, h)

            # (e) Drena resultados de DeepFace prontos
            for pid, res, err in emotion_worker.get_results():
                if err is not None:
                    erros_deepface += 1
                    if erros_deepface <= 3 or erros_deepface % 50 == 0:
                        logger.warning(f"Falha na análise emocional ({erros_deepface}): {err}")
                    continue
                if pid not in tracker.estados:
                    continue
                estado = tracker.estados[pid]
                try:
                    emotion_raw = res[0]['dominant_emotion']
                    confidence = res[0]['emotion'][emotion_raw]
                    if confidence < 38:
                        emotion_raw = 'neutral'
                        confidence = res[0]['emotion']['neutral']
                    elif emotion_raw == 'surprise' and confidence < 75:
                        emotion_raw = 'neutral'
                        confidence = res[0]['emotion']['neutral']

                    estado.emotion_buffer.append(emotion_raw)
                    emo = Counter(estado.emotion_buffer).most_common(1)[0][0]

                    estado.last_label, estado.last_color = EMOCAO_MAP.get(
                        emo, (emo.capitalize(), (200, 200, 200))
                    )
                    estado.last_confidence = confidence
                    estado.emotion_counter[emo] += 1

                    wellness_atual = calcular_wellness(estado.emotion_counter)
                    burnout_atual = calcular_risco_burnout(
                        estado.emotion_counter,
                        estado.alertas_fadiga_total,
                        estado.heart_rate_bpm,
                        estado.heart_rate_qualidade,
                        estado.bpm_baseline,
                        estado.perclos_atual,
                    )
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    buffer_csv.append([
                        ts,
                        f"P{pid}",
                        emo,
                        f"{confidence:.2f}",
                        f"{wellness_atual:.1f}",
                        f"{burnout_atual:.1f}",
                        f"{estado.heart_rate_bpm:.1f}" if estado.heart_rate_bpm is not None else "",
                        f"{estado.heart_rate_qualidade:.3f}" if estado.heart_rate_qualidade is not None else "",
                        f"{estado.perclos_atual:.1f}",
                        f"{estado.ear_threshold:.3f}",
                    ])

                    # (b) Alerta com histerese
                    if estado.avaliar_alerta_negativo(emo):
                        tocar_alerta()
                except Exception as e:
                    erros_deepface += 1
                    if erros_deepface <= 3 or erros_deepface % 50 == 0:
                        logger.warning(f"Erro processando resultado DeepFace: {e}")

            # ─── Renderização ───
            for indice_pid, (pid, (x, y, w, h)) in enumerate(zip(ids, faces)):
                estado = tracker.estados[pid]
                face_roi = frame[y:y+h, x:x+w]

                if modo_privacidade and face_roi.size > 0:
                    frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (55, 55), 30)

                cor = estado.last_color
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)

                label = f"P{pid}: {estado.last_label}"
                if estado.last_confidence is not None:
                    label += f" ({estado.last_confidence:.0f}%)"

                # Indicador de calibração
                if not estado.ear_calibrado:
                    label += "  [CALIB EAR...]"
                if estado.bpm_baseline is None and estado.heart_rate_bpm is not None:
                    label += "  [CALIB BPM...]"

                cv2.putText(frame, label, (x, max(y - 10, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

                cv2.putText(frame, f"Piscadas: {estado.piscadas} | PERCLOS: {estado.perclos_atual:.0f}%",
                            (x, min(y + h + 18, h_frame - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)

                if estado.heart_rate_bpm is not None:
                    cor_hr = (0, 220, 220)
                    if estado.heart_rate_qualidade is not None and estado.heart_rate_qualidade < RPPG_QUALIDADE_MIN:
                        cor_hr = (120, 120, 120)
                    base_txt = f" (base {estado.bpm_baseline:.0f})" if estado.bpm_baseline else ""
                    cv2.putText(frame, f"FC: {estado.heart_rate_bpm:.0f} bpm{base_txt}",
                                (x, min(y + h + 54, h_frame - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_hr, 1)

                risco_b = calcular_risco_burnout(
                    estado.emotion_counter,
                    estado.alertas_fadiga_total,
                    estado.heart_rate_bpm,
                    estado.heart_rate_qualidade,
                    estado.bpm_baseline,
                    estado.perclos_atual,
                )
                nivel_b, cor_b = classificar_burnout(risco_b)
                cv2.putText(frame, f"Burnout: {nivel_b} ({risco_b:.0f}%)",
                            (x, min(y + h + 36, h_frame - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_b, 1)

                if estado.alerta_ativo:
                    overlay = frame.copy()
                    cy = h_frame // 2
                    cv2.rectangle(overlay, (0, cy - 45), (w_frame, cy + 45), (0, 0, 160), -1)
                    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                    cv2.putText(frame, "PAUSA RECOMENDADA  —  Respire fundo!",
                                (20, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

                # (i) Usa indice_pid (enumerate) em vez de list().index()
                if estado.alerta_fadiga:
                    cv2.putText(frame, f"P{pid}: FADIGA — piscadas elevadas",
                                (10, 60 + indice_pid * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                if estado.perclos_atual > LIMIAR_PERCLOS_ALERTA:
                    cv2.putText(frame, f"P{pid}: SONOLENCIA — PERCLOS {estado.perclos_atual:.0f}%",
                                (10, 85 + indice_pid * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            # HUD
            modo_txt = "  [PRIVACIDADE ATIVA]" if modo_privacidade else ""
            cv2.putText(frame, f"Monitoramento Ativo - SBRT 2026{modo_txt}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "q = sair   |   p = modo privacidade",
                        (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("Sistema de Saude Mental - Teletrabalho Ibmec", frame)

            if len(buffer_csv) >= CSV_BUFFER_SIZE:
                salvar_registros_csv(buffer_csv)
                buffer_csv = []

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                break
            elif tecla == ord('p'):
                modo_privacidade = not modo_privacidade

    except KeyboardInterrupt:
        print("\n⏹ Monitoramento interrompido pelo usuário")
    finally:
        if buffer_csv:
            salvar_registros_csv(buffer_csv)

        emotion_worker.stop()
        cap.release()
        if face_mesh is not None:
            face_mesh.close()
        cv2.destroyAllWindows()

        emotion_global = tracker.emotion_counter_global()
        wellness_final = calcular_wellness(emotion_global)
        bpm_baseline_final = tracker.bpm_baseline_medio()
        hr_medio = tracker.heart_rate_medio()
        perclos_medio = tracker.perclos_medio()
        burnout_final = calcular_risco_burnout(
            emotion_global,
            tracker.alertas_fadiga_total(),
            hr_medio,
            None,                 # qualidade desconhecida no agregado
            bpm_baseline_final,
            perclos_medio,
        )
        gerar_relatorio(emotion_global, wellness_final, tracker.piscadas_total(),
                        burnout_final, hr_medio, perclos_medio)


# ─── Variável global para fallback Haar (preenchida no main) ────────────────────
eye_cascade_fallback = None


if __name__ == "__main__":
    main()