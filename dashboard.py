import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import time

LOG_FILE = "monitoramento_emocional.csv"

EMOCAO_MAP = {
    'happy': 'Feliz', 'neutral': 'Neutro', 'sad': 'Triste',
    'angry': 'Raiva', 'fear': 'Medo', 'disgust': 'Desgosto', 'surprise': 'Surpresa',
}

PESOS_WELLNESS = {
    'happy': 1.0, 'neutral': 0.0, 'surprise': 0.2,
    'sad': -0.6, 'fear': -0.7, 'disgust': -0.5, 'angry': -0.8,
}

COR_EMOCAO = {
    'happy': '#00DC00', 'neutral': '#AAAAAA', 'sad': '#DC5000',
    'angry': '#4444FF', 'fear': '#008CFF', 'disgust': '#009090', 'surprise': '#FFD600',
}

MAPA_NUMERICO = {
    'happy': 5, 'surprise': 4, 'neutral': 3,
    'fear': 2, 'disgust': 2, 'sad': 1, 'angry': 0,
}

st.set_page_config(
    page_title="Monitor Emocional — Teletrabalho",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Monitor Emocional — Teletrabalho")
st.caption("Ibmec · SBRT 2026")


@st.cache_data(ttl=3)
def carregar_dados():
    if not os.path.isfile(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    except Exception:
        return None

    if 'Emotion' not in df.columns and len(df.columns) >= 2:
        df.columns = (['Timestamp', 'Emotion'] + list(df.columns[2:]))[:len(df.columns)]

    if df.empty or 'Emotion' not in df.columns:
        return None

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp', 'Emotion'])
    df['Emotion_PT'] = df['Emotion'].map(EMOCAO_MAP)

    if 'PessoaID'   not in df.columns: df['PessoaID']   = 'P0'
    if 'Confidence' not in df.columns: df['Confidence'] = None
    if 'Wellness'   not in df.columns: df['Wellness']   = None

    if df['Wellness'] is not None:
        df['Wellness'] = pd.to_numeric(df['Wellness'], errors='coerce')

    return df


df = carregar_dados()

if df is None:
    st.warning("⏳ Aguardando dados do monitoramento... Inicie o aplicativo de captura.")
    st.info("O dashboard atualiza automaticamente a cada 3 segundos.")
    time.sleep(3)
    st.rerun()

# ── Métricas ─────────────────────────────────────────────────────────────────
total = len(df)
soma_pesos = df['Emotion'].map(PESOS_WELLNESS).fillna(0).sum()
wellness = round(max(0.0, min(100.0, 50.0 + (soma_pesos / total) * 50.0)), 1)
emo_dominante = df['Emotion'].value_counts().idxmax()

corte = pd.Timestamp.now() - pd.Timedelta(seconds=30)
pessoas = max(df[df['Timestamp'] >= corte]['PessoaID'].nunique(), 1)

wellness_status  = "Bom"      if wellness >= 70 else "Atenção" if wellness >= 40 else "Crítico"
wellness_d_color = "normal"   if wellness >= 70 else "off"     if wellness >= 40 else "inverse"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total de análises", total)
with c2:
    st.metric("Emoção dominante", EMOCAO_MAP.get(emo_dominante, emo_dominante))
with c3:
    st.metric("Wellness Score", f"{wellness} / 100",
              delta=wellness_status, delta_color=wellness_d_color)
with c4:
    st.metric("Pessoas ativas (30 s)", pessoas)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Gráficos", "👥  Por Pessoa", "📋  Registros"])

with tab1:
    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Distribuição de emoções")
        contagem = (
            df['Emotion'].value_counts()
            .rename_axis('Emotion')
            .reset_index(name='count')
        )
        contagem['Emoção'] = contagem['Emotion'].map(EMOCAO_MAP)
        fig1 = px.pie(
            contagem,
            values='count',
            names='Emoção',
            color='Emotion',
            color_discrete_map=COR_EMOCAO,
            hole=0.35,
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig1, width='content')

    with col_dir:
        st.subheader("Linha do tempo emocional")
        df_recente = df.tail(300).copy()
        df_recente['Valor']     = df_recente['Emotion'].map(MAPA_NUMERICO).fillna(3)
        df_recente['Emoção']    = df_recente['Emotion'].map(EMOCAO_MAP)

        fig2 = go.Figure()
        fig2.add_hrect(y0=0, y1=2.5, fillcolor="rgba(220,80,0,0.06)", line_width=0)
        fig2.add_hrect(y0=2.5, y1=5.5, fillcolor="rgba(0,180,0,0.05)", line_width=0)
        fig2.add_trace(go.Scatter(
            x=df_recente['Timestamp'],
            y=df_recente['Valor'],
            mode='lines',
            line=dict(color='#4a90d9', width=2),
            hovertemplate='%{x|%H:%M:%S}<br>%{text}<extra></extra>',
            text=df_recente['Emoção'],
        ))
        fig2.update_layout(
            yaxis=dict(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=['Raiva', 'Triste', 'Medo/Desgosto', 'Neutro', 'Surpresa', 'Feliz'],
                range=[-0.3, 5.3],
                gridcolor='rgba(200,200,200,0.3)',
            ),
            xaxis=dict(title="Tempo", gridcolor='rgba(200,200,200,0.2)'),
            margin=dict(t=10, b=40, l=110, r=10),
            height=320,
            showlegend=False,
            plot_bgcolor='white',
        )
        fig2.add_hline(y=3, line_dash="dot", line_color="gray", opacity=0.4,
                       annotation_text="Neutro", annotation_position="right")
        st.plotly_chart(fig2, width='content')

with tab2:
    df_w = df[df['Wellness'].notna()].copy()
    if df_w.empty:
        st.info("Dados de wellness por pessoa ainda não disponíveis.")
    else:
        df_ultima = df_w.groupby('PessoaID').last().reset_index()
        for _, row in df_ultima.iterrows():
            try:
                w = float(row['Wellness'])
            except (ValueError, TypeError):
                continue
            if not (0 <= w <= 100):
                continue
            icone = "🟢" if w >= 70 else "🟡" if w >= 40 else "🔴"
            col_a, col_b = st.columns([1, 5])
            with col_a:
                st.markdown(f"**{row['PessoaID']}** {icone}")
                st.caption(f"{w:.1f} / 100")
            with col_b:
                st.progress(int(w))

with tab3:
    st.subheader("Últimas 20 análises")
    df_show = df.tail(20)[['Timestamp', 'PessoaID', 'Emotion_PT', 'Confidence', 'Wellness']].copy()
    df_show.columns = ['Horário', 'Pessoa', 'Emoção', 'Confiança (%)', 'Wellness']
    st.dataframe(df_show.iloc[::-1], width='content', hide_index=True)

st.divider()
st.caption(
    f"Atualizado às {datetime.datetime.now().strftime('%H:%M:%S')} "
    "— atualiza automaticamente a cada 3 s"
)

time.sleep(3)
st.rerun()
