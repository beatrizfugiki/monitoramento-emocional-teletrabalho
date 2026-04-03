import cv2
from deepface import DeepFace
from collections import Counter
import pandas as pd
import datetime
import os

# --- Configurações Iniciais ---

# Define o caminho do modelo Haar Cascade (algoritmo clássico para detectar rostos em imagens)
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Define o nome do arquivo CSV onde os dados serão acumulados
log_file = "monitoramento_emocional.csv"

# Verifica se o arquivo CSV já existe; se não, cria um novo com o cabeçalho (Timestamp e Emoção)
if not os.path.isfile(log_file):
    df_init = pd.DataFrame(columns=['Timestamp', 'Emotion'])
    df_init.to_csv(log_file, index=False)

# Inicializa a captura de vídeo da webcam padrão (índice 0)
cap = cv2.VideoCapture(0)

# Inicializa um contador para somar quantas vezes cada emoção foi detectada na sessão
emotion_counter = Counter()

# Contador de frames para controlar a frequência da análise de IA
frame_count = 0

# Variável para armazenar o texto da última emoção detectada com sucesso
last_emotion = "Iniciando..."

print("Iniciando Monitoramento... Pressione 'q' para sair.")

# Loop principal de processamento de vídeo
while True:
    # Captura frame por frame da webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Incrementa o contador de frames
    frame_count += 1
    
    # Converte a imagem colorida (BGR) para tons de cinza (mais rápido para detecção de rostos)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostos na imagem em tons de cinza
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    # Itera sobre cada rosto encontrado no frame
    for (x, y, w, h) in faces:
        
        # 1. Feedback Visual: Desenha um retângulo verde ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 2. Performance: A análise de IA é pesada, por isso só rodamos a cada 10 frames (~3x por segundo)
        if frame_count % 10 == 0:
            # Recorta apenas a área do rosto (Region of Interest) para enviar à IA
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # DeepFace.analyze: Chama a rede neural para classificar a expressão facial
                # enforce_detection=False evita erros se o rosto estiver parcialmente oculto
                res = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                
                # Extrai o nome da emoção dominante (ex: 'happy', 'neutral', 'sad')
                emotion_label = res[0]['dominant_emotion']
                last_emotion = emotion_label
                
                # 3. Registro de Dados: Captura o horário exato da detecção
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Cria uma linha de dados e anexa (append) ao arquivo CSV sem sobrescrever o anterior
                new_data = pd.DataFrame([[timestamp, emotion_label]], columns=['Timestamp', 'Emotion'])
                new_data.to_csv(log_file, mode='a', header=False, index=False)
                
                # Atualiza o contador de estatísticas da sessão atual
                emotion_counter[emotion_label] += 1

            except Exception as e:
                # Se a IA falhar (luz ruim, ângulo difícil), exibe o erro no console e na tela
                print(f"Erro na análise: {e}")
                last_emotion = "Erro"

        # 4. Interface de Usuário: Escreve o nome da emoção acima do retângulo do rosto
        cv2.putText(frame, f"Emocao: {last_emotion}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. Adiciona um rótulo fixo no topo da imagem para identificação do projeto
    cv2.putText(frame, "Monitoramento Ativo - SBRT 2026", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Abre a janela e exibe o vídeo resultante
    cv2.imshow("Sistema de Preservacao de Saude Mental - Ibmec", frame)

    # Interrompe o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza a captura e fecha todas as janelas abertas
cap.release()
cv2.destroyAllWindows()

# --- Relatório Final ---
# Imprime no console um resumo estatístico da sessão de teletrabalho
print("\n" + "="*40)
print("RELATÓRIO DE SESSÃO (TELETRABALHO)")
print("="*40)

# Calcula o total de frames analisados para converter os números em porcentagem
total_analises = sum(emotion_counter.values())

if total_analises > 0:
    for emotion, count in emotion_counter.items():
        percent = (count / total_analises) * 100
        print(f"{emotion.capitalize():<15}: {count:<5} ({percent:.2f}%)")

print("="*40)
print(f"Os dados detalhados foram salvos em: {log_file}")