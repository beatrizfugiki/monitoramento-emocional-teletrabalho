import cv2
from deepface import DeepFace
from collections import Counter, deque
import pandas as pd
import datetime
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform

# ─── Configurações ──────────────────────────────────────────────────────────────

LOG_FILE = "monitoramento_emocional.csv"
LIMITE_ALERTA_SEGUNDOS = 180
FRAMES_PISCAR = 3
LIMIAR_PISCADAS_MIN = 25
CSV_BUFFER_SIZE = 32  # Escreve a cada 32 registros (≈4 segundos)

# Limiares de risco de burnout (0-100)
LIMIAR_BURNOUT_BAIXO    = 25
LIMIAR_BURNOUT_MODERADO = 50
LIMIAR_BURNOUT_ALTO     = 75

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

PESOS_WELLNESS = {
    'happy': 1.0, 'neutral': 0.0, 'surprise': 0.2,
    'sad': -0.6, 'fear': -0.7, 'disgust': -0.5, 'angry': -0.8,
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

# ─── CSV ────────────────────────────────────────────────────────────────────────

COLUNAS_CSV = ['Timestamp', 'PessoaID', 'Emotion', 'Confidence', 'Wellness', 'BurnoutRisk']

def inicializar_csv():
    if os.path.isfile(LOG_FILE):
        try:
            cabecalho = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
            if cabecalho != COLUNAS_CSV:
                backup = LOG_FILE.replace('.csv', '_backup.csv')
                os.rename(LOG_FILE, backup)
                print(f"Formato antigo salvo em: {backup}")
                pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)
        except Exception as e:
            print(f"Erro ao validar CSV: {e}")
            pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)
    else:
        pd.DataFrame(columns=COLUNAS_CSV).to_csv(LOG_FILE, index=False)

inicializar_csv()

# Inicializa câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Câmera não encontrada. Verifique a conexão.")
    exit(1)

# ─── Funções auxiliares ──────────────────────────────────────────────────────────

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

def calcular_wellness(counter):
    total = sum(counter.values())
    if total == 0:
        return 50.0
    soma = sum(PESOS_WELLNESS.get(e, 0) * c for e, c in counter.items())
    return round(max(0.0, min(100.0, 50.0 + (soma / total) * 50.0)), 1)

def calcular_risco_burnout(counter, alertas_fadiga=0):
    """Índice de risco de burnout (0-100).
    Componentes: 60% emoções negativas + 30% wellness invertido + 10% fadiga ocular.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    neg = sum(counter.get(e, 0) for e in EMOCOES_NEGATIVAS)
    pontos_emocao   = (neg / total) * 60.0
    pontos_wellness = max(0.0, (100.0 - calcular_wellness(counter)) / 100.0 * 30.0)
    pontos_fadiga   = min(alertas_fadiga * 5.0, 10.0)
    return round(min(100.0, pontos_emocao + pontos_wellness + pontos_fadiga), 1)

def classificar_burnout(risco):
    if risco < LIMIAR_BURNOUT_BAIXO:    return "Baixo",    (0, 200, 0)
    if risco < LIMIAR_BURNOUT_MODERADO: return "Moderado", (0, 165, 255)
    if risco < LIMIAR_BURNOUT_ALTO:     return "Alto",     (30, 100, 255)
    return "Crítico", (0, 0, 220)

def detectar_olhos(face_roi):
    """Retorna True se ao menos um olho for detectado."""
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        regiao_olhos = gray[:int(h * 0.6), :]
        olhos = eye_cascade.detectMultiScale(regiao_olhos, 1.1, 3, minSize=(15, 15))
        return len(olhos) > 0
    except Exception:
        return True  # Assume que viu olho se houver erro

def salvar_registros_csv(registros):
    """Escreve múltiplos registros de uma vez"""
    if registros:
        try:
            pd.DataFrame(registros, columns=COLUNAS_CSV).to_csv(
                LOG_FILE, mode='a', header=False, index=False
            )
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")

def gerar_relatorio(emotion_counter, wellness, piscadas_total, burnout_risk=0.0):
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
    print("=" * 47)
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

# ─── Estado por pessoa ───────────────────────────────────────────────────────────

class EstadoPessoa:
    def __init__(self, pid):
        self.pid = pid
        self.emotion_buffer = deque(maxlen=5)
        self.emotion_counter = Counter()
        self.last_label = "Analisando..."
        self.last_color = (200, 200, 200)
        self.last_confidence = None
        self.inicio_negativa = None
        self.alerta_ativo = False
        
        # Fadiga
        self.frames_sem_olho = 0
        self.piscadas = 0
        self.piscadas_janela = 0
        self.inicio_janela = datetime.datetime.now()
        self.alerta_fadiga = False
        self.alerta_fadiga_timestamp = None
        self.alertas_fadiga_total = 0

    def atualizar_piscada(self, olho_detectado):
        """Atualiza contagem de piscadas e detecta fadiga"""
        if not olho_detectado:
            self.frames_sem_olho += 1
        else:
            if self.frames_sem_olho >= FRAMES_PISCAR:
                self.piscadas += 1
                self.piscadas_janela += 1
            self.frames_sem_olho = 0

        agora = datetime.datetime.now()
        # Usa total_seconds() em vez de .seconds
        if (agora - self.inicio_janela).total_seconds() >= 60:
            if self.piscadas_janela > LIMIAR_PISCADAS_MIN:
                self.alerta_fadiga = True
                self.alerta_fadiga_timestamp = agora
                self.alertas_fadiga_total += 1
            self.piscadas_janela = 0
            self.inicio_janela = agora
        
        # Reset do alerta de fadiga após 30 segundos
        if (self.alerta_fadiga and self.alerta_fadiga_timestamp and 
            (agora - self.alerta_fadiga_timestamp).total_seconds() >= 30):
            self.alerta_fadiga = False

# ─── Tracker multi-rosto ────────────────────────────────────────────────────────

class TrackerRosto:
    def __init__(self):
        self.proxima_id = 0
        self.ids_centros = {}
        self.estados = {}
        self.frame_visto = {}

    def atualizar(self, centros, frame_count, max_ausente=90):
        for rid in list(self.ids_centros):
            if frame_count - self.frame_visto.get(rid, 0) > max_ausente:
                del self.ids_centros[rid]

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
        total = Counter()
        for e in self.estados.values():
            total += e.emotion_counter
        return total

    def piscadas_total(self):
        return sum(e.piscadas for e in self.estados.values())

    def alertas_fadiga_total(self):
        return sum(e.alertas_fadiga_total for e in self.estados.values())

# ─── Loop principal ──────────────────────────────────────────────────────────────

tracker = TrackerRosto()
frame_count = 0
modo_privacidade = False
buffer_csv = []  # Buffer para escrever CSV em lotes

print("Iniciando Monitoramento...  |  'q' = sair  |  'p' = modo privacidade")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERRO: Não foi possível ler frame da câmera.")
            break

        frame_count += 1
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        centros = [((x + w // 2), (y + h // 2)) for (x, y, w, h) in faces]
        ids = tracker.atualizar(centros, frame_count)

        for pid, (x, y, w, h) in zip(ids, faces):
            estado = tracker.estados[pid]
            face_roi = frame[y:y+h, x:x+w]

            olho_detectado = detectar_olhos(face_roi) if face_roi.size > 0 else True
            estado.atualizar_piscada(olho_detectado)

            # Análise a cada 8 frames
            if frame_count % 8 == 0 and face_roi.size > 0:
                try:
                    res = DeepFace.analyze(
                        face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='skip',
                    )
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
                    burnout_atual = calcular_risco_burnout(estado.emotion_counter, estado.alertas_fadiga_total)
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Adiciona ao buffer em vez de escrever imediatamente
                    buffer_csv.append([ts, f"P{pid}", emo, f"{confidence:.2f}", f"{wellness_atual:.1f}", f"{burnout_atual:.1f}"])

                    # Alerta de emoção negativa (usa total_seconds())
                    if emo in EMOCOES_NEGATIVAS:
                        if estado.inicio_negativa is None:
                            estado.inicio_negativa = datetime.datetime.now()
                        elif (datetime.datetime.now() - estado.inicio_negativa).total_seconds() >= LIMITE_ALERTA_SEGUNDOS:
                            if not estado.alerta_ativo:
                                estado.alerta_ativo = True
                                tocar_alerta()
                    else:
                        estado.inicio_negativa = None
                        estado.alerta_ativo = False

                except Exception as e:
                    pass  # Silencia erros do DeepFace

            # Renderização
            if modo_privacidade and face_roi.size > 0:
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (55, 55), 30)

            cor = estado.last_color
            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)

            label = f"P{pid}: {estado.last_label}"
            if estado.last_confidence is not None:
                label += f" ({estado.last_confidence:.0f}%)"
            cv2.putText(frame, label, (x, max(y - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

            cv2.putText(frame, f"Piscadas: {estado.piscadas}",
                        (x, min(y + h + 18, h_frame - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)

            risco_b = calcular_risco_burnout(estado.emotion_counter, estado.alertas_fadiga_total)
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

            if estado.alerta_fadiga:
                cv2.putText(frame, f"P{pid}: FADIGA — piscadas elevadas",
                            (10, 60 + list(tracker.estados.keys()).index(pid) * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # HUD
        modo_txt = "  [PRIVACIDADE ATIVA]" if modo_privacidade else ""
        cv2.putText(frame, f"Monitoramento Ativo — SBRT 2026{modo_txt}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "q = sair   |   p = modo privacidade",
                    (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Sistema de Saude Mental — Teletrabalho Ibmec", frame)

        # Escreve CSV em lotes
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
    # Salva registros restantes no buffer
    if buffer_csv:
        salvar_registros_csv(buffer_csv)
    
    cap.release()
    cv2.destroyAllWindows()

    # ─── Relatório Final ─────────────────────────────────────────────────────────────
    emotion_global = tracker.emotion_counter_global()
    wellness_final = calcular_wellness(emotion_global)
    burnout_final  = calcular_risco_burnout(emotion_global, tracker.alertas_fadiga_total())
    gerar_relatorio(emotion_global, wellness_final, tracker.piscadas_total(), burnout_final)