# Monitoramento Emocional no Teletrabalho

Este projeto usa visão computacional para acompanhar emoções em tempo real pela webcam, registrar os resultados em um arquivo CSV e exibir um painel analítico com indicadores de bem-estar e risco de burnout.

## Visão geral

O repositório tem duas partes principais:

1. `projeto-monitoramento-emocional-teletrabalho.py` faz a captura ao vivo pela webcam, detecta rostos, identifica emoção, estima wellness e burnout e salva tudo em `monitoramento_emocional.csv`.
2. `dashboard.py` lê esse CSV e mostra os dados em um painel com métricas, gráficos e histórico por pessoa.

A ideia central é simples: o script de captura gera os dados e o dashboard consome esses dados para análise visual.

## Como o código funciona

### 1. Captura e análise em tempo real

O arquivo `projeto-monitoramento-emocional-teletrabalho.py` é o motor do sistema. Ele:

- abre a câmera com OpenCV;
- detecta rostos com classificadores Haar;
- usa a biblioteca DeepFace para classificar a emoção dominante;
- suaviza decisões com um pequeno buffer de emoções por pessoa;
- calcula um score de wellness com base no peso de cada emoção;
- estima risco de burnout combinando emoções negativas, wellness e fadiga ocular;
- grava cada leitura no CSV.

Além disso, o script mantém estado por pessoa com um rastreador simples de centros faciais, para tentar acompanhar indivíduos diferentes ao longo do tempo.

### 2. Dashboard analítico

O arquivo `dashboard.py` é a camada visual. Ele:

- carrega o CSV atualizado;
- converte timestamps e prepara colunas auxiliares;
- calcula métricas gerais, como total de análises, emoção dominante, wellness e burnout;
- mostra gráficos de distribuição emocional e linha do tempo;
- exibe a visão por pessoa;
- apresenta as últimas leituras em tabela.

O painel se atualiza automaticamente a cada 3 segundos.

## Explicação das principais partes do código

### Constantes e mapeamentos

Os dois arquivos usam tabelas de apoio para padronizar o sistema:

- `EMOCAO_MAP`: traduz emoções em inglês para português;
- `PESOS_WELLNESS`: define o peso de cada emoção no cálculo do bem-estar;
- `EMOCOES_NEGATIVAS`: agrupa emoções que aumentam o risco emocional;
- `LIMIAR_BURNOUT_*`: define faixas de interpretação do risco;
- `MAPA_NUMERICO`: transforma emoções em uma escala numérica para os gráficos;
- `COR_EMOCAO`: define as cores usadas na visualização.

Essas estruturas evitam repetir valores espalhados pelo código e deixam as regras mais fáceis de ajustar.

### Cálculo de wellness e burnout

O wellness parte de um valor central e é ajustado conforme o peso das emoções observadas. Emoções positivas elevam o score; emoções negativas reduzem.

O burnout usa uma combinação de fatores:

- proporção de emoções negativas;
- wellness invertido;
- fadiga ocular, quando aplicável.

Isso gera um índice entre 0 e 100, depois classificado como baixo, moderado, alto ou crítico.

### Persistência em CSV

O arquivo CSV é o elo entre captura e dashboard. Cada linha salva pelo script de coleta contém:

- `Timestamp`
- `PessoaID`
- `Emotion`
- `Confidence`
- `Wellness`
- `BurnoutRisk`

O dashboard tolera arquivos antigos ou com colunas faltando, preenchendo valores padrão quando necessário.

### Visualização no Streamlit

O dashboard usa três abas:

- **Gráficos**: mostra a distribuição das emoções, a linha do tempo emocional e a evolução do burnout;
- **Por Pessoa**: apresenta o último estado conhecido de cada pessoa;
- **Registros**: exibe as análises mais recentes em tabela.

## Estrutura do repositório

- `projeto-monitoramento-emocional-teletrabalho.py`: captura e processamento em tempo real.
- `dashboard.py`: painel analítico em Streamlit.
- `README.md`: documentação do projeto.

## Requisitos

O projeto depende principalmente de:

- `opencv-python`
- `deepface`
- `pandas`
- `numpy`
- `matplotlib`
- `streamlit`
- `plotly`

Se faltar alguma dependência no ambiente, instale com `pip`.

## Como executar

1. Instale as dependências do projeto.
2. Rode o script de captura para começar a gerar o CSV.
3. Em outro terminal, execute o dashboard.

Exemplo:

```bash
python projeto-monitoramento-emocional-teletrabalho.py
streamlit run dashboard.py
```

## Observações importantes

- O script de captura exige acesso à webcam.
- O dashboard só mostra dados quando o CSV já tiver registros.
- Se o CSV for apagado, o sistema volta a preencher o arquivo a partir da próxima execução da captura.

## Objetivo do projeto

O propósito é oferecer uma visão contínua do estado emocional de colaboradores em teletrabalho, permitindo identificar padrões de alerta e apoiar decisões voltadas ao bem-estar.
