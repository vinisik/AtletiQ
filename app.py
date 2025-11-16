import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import time

# -----------------------------------
# IMPORTS DOS SEUS MÓDULOS LOCAIS
# -----------------------------------
try:
    from web_scraper import buscar_dados_brasileirao
    from feature_engineering import preparar_dados_para_modelo
    from model_trainer import treinar_modelo
    from predictor import prever_jogo_especifico, simular_campeonato
    from analysis import gerar_confronto_direto
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}. Verifique se todos os arquivos .py (web_scraper, analysis, etc.) estão na mesma pasta.")
    st.stop()


st.set_page_config(
    page_title="AtletiQ | Estatísticas do Brasileirão",
    page_icon="⚽",
    layout="wide"
)

def get_base64_image(image_path):
    """Converte imagem local em base64 para ser exibida via HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Arquivo de logo não encontrado: {image_path}")
        return None

# Carrega o logo
logo_base64 = get_base64_image("logo.png")

if logo_base64:
    logo_css = f"""
        content: ""; 
        background-image: url("data:image/png;base64,{logo_base64}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        width: 150px;  
        height: 75px;
    """
else:
    logo_css = f"""
        content: "AtletiQ"; /* Fallback de texto */
        font-size: 1.5rem;
        color: white;
        font-weight: bold;
        padding-top: 5px; 
    """

st.markdown(f"""
    <style>
        div[data-testid="stSidebarNav"] {{
            display: none;
        }}
        div[data-testid="stSidebar"] {{
            display: none;
        }}
        button[title="View app source"] {{
            display: none;
        }}
        
        .block-container {{
            padding-top: 2.3rem;
        }}

        div[data-baseweb="tab-list"] {{
            display: flex; 
            justify-content: center !important; 
            align-items: center; 
            position: relative; 
            width: 100%;
            border-bottom: 2px solid #333;
            min-height: 95px; 
        }}

        div[data-baseweb="tab-list"]::before {{
            {logo_css}

            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%); 
        }}

        button[data-baseweb="tab"] {{
            background-color: transparent !important;
            color: #AAA !important; 
            border: none !important;
            margin: 0 15px !important;
            font-weight: 500;
            font-size: 1.05rem;
            transition: color 0.2s;
        }}
        
        button[data-baseweb="tab"]:hover {{
            color: #FFF !important; 
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: white !important; 
            font-weight: 600;
        }}

        .block-container {{
            padding-top: 2rem;
            
            
            max-width: 1400px;  /
            margin: 0 auto;     
        }}

        div[data-baseweb="tab-highlight"] {{
            background-color: #00C853 !important; 
            height: 3px !important;
            border-radius: 3px 3px 0 0;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados_e_modelo():
    with st.spinner('Baixando dados da temporada... Isso pode levar um momento.'):
        ano_atual = datetime.now().year
        df_total = buscar_dados_brasileirao(str(ano_atual))
        if df_total is None or df_total.empty:
             st.warning(f"Não foi possível buscar dados de {ano_atual}. Tentando ano anterior (2024)...")
             df_total = buscar_dados_brasileirao(str(ano_atual - 1))

    if df_total is None or df_total.empty:
        st.error("Falha ao buscar os dados. Tente recarregar a página.")
        return (None,) * 8 # Retorna 8 Nones

    df_resultados = df_total[df_total['FTHG'].notna()].copy()
    df_futuro = df_total[df_total['FTHG'].isna()].copy()

    with st.spinner(f'Calculando features e treinando o modelo com dados...'):
        df_treino, time_stats = preparar_dados_para_modelo(df_resultados)
        
        if df_treino.empty or len(df_treino) < 10: 
            st.warning("Ainda não há dados de treino suficientes na temporada para treinar um modelo.")
            lista_times = sorted(list(set(df_total['HomeTeam']).union(set(df_total['AwayTeam']))))
            return df_resultados, df_futuro, None, None, None, None, lista_times, df_total
        
        modelo, encoder, colunas_modelo = treinar_modelo(df_treino)

    lista_times = sorted(list(set(df_total['HomeTeam']).union(set(df_total['AwayTeam']))))
    return df_resultados, df_futuro, time_stats, modelo, encoder, colunas_modelo, lista_times, df_total


(df_resultados, df_futuro, time_stats, 
 modelo, encoder, colunas_modelo, 
 lista_times, df_total) = carregar_dados_e_modelo()



if lista_times:
    
    tab_previsao, tab_simulacao, tab_confronto = st.tabs(
        ["Previsão de Jogo", "Simulação da Tabela", "Confronto Direto"]
    )


    # ----- Aba de Previsão de Jogo -----
    with tab_previsao:
        if modelo is None or time_stats is None:
            st.error("Não há dados suficientes para treinar o modelo de previsão ainda. Tente novamente mais tarde na temporada.")
        else:
            st.header("Previsão de Jogo Específico", anchor=None) 
            st.markdown("Selecione dois times para ver as probabilidades de vitória, empate ou derrota para a partida.")
            
            col1, col2 = st.columns(2)
            with col1:
                time_casa = st.selectbox("Time da casa:", lista_times, index=None, key="select_casa", placeholder="Escolha o time da casa")
            with col2:
                time_visitante = st.selectbox("Time visitante:", lista_times, index=None, key="select_visitante", placeholder="Escolha o time visitante")
                
            if st.button("Prever Resultado", use_container_width=False, type="primary", key="btn_prever"):
                if time_casa and time_visitante:
                    if time_casa == time_visitante:
                        st.warning("O time da casa e o visitante devem ser diferentes.")
                    else:
                        with st.spinner('Calculando probabilidades...'):
                            odds = prever_jogo_especifico(time_casa, time_visitante, modelo, encoder, time_stats, colunas_modelo)
                        
                        st.subheader(f"Probabilidades: {time_casa} vs {time_visitante}")
                        
                        odds_display = {
                            'Vitória Casa': odds.get('Vitória Casa', 0),
                            'Empate': odds.get('Empate', 0),
                            'Vitória Visitante': odds.get('Vitória Visitante', 0)
                        }

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            prob_casa = odds_display['Vitória Casa']
                            st.metric(f"Vitória {time_casa}", f"{prob_casa:.1%}")
                        with c2:
                            prob_empate = odds_display['Empate']
                            st.metric("Empate", f"{prob_empate:.1%}")
                        with c3:
                            prob_visitante = odds_display['Vitória Visitante']
                            st.metric(f"Vitória {time_visitante}", f"{prob_visitante:.1%}")
                else:
                    st.error("Por favor, selecione os dois times para fazer a previsão.")

    # ----- Aba de Simulação do Campeonato -----
    with tab_simulacao:
        if modelo is None or time_stats is None or df_resultados is None or df_futuro is None:
            st.error("Não há dados suficientes para simular o campeonato ainda. Tente novamente mais tarde na temporada.")
        else:
            st.header("Simulação da Tabela", anchor=None)
            st.markdown("Simule os resultados futuros e veja a provável classificação final do campeonato.")
            
            rodada_atual = 0
            if not df_resultados.empty and 'Rodada' in df_resultados.columns:
                rodadas_validas = pd.to_numeric(df_resultados['Rodada'], errors='coerce').dropna()
                if not rodadas_validas.empty:
                    rodada_atual = int(rodadas_validas.max())
            
            rodada_simulacao = st.slider(f"Simular até qual rodada? (Rodada atual: {rodada_atual})",
                                         min_value=max(1, rodada_atual),
                                         max_value=38,
                                         value=38)
            
            if st.button("Simular Tabela", use_container_width=False, type="primary", key="btn_simular"):
                if df_futuro.empty and rodada_atual >= 38:
                    st.info("O campeonato já terminou! Exibindo a tabela final.")
                    tabela_final = simular_campeonato(38, df_futuro, df_resultados, modelo, encoder, time_stats, colunas_modelo)
                    st.dataframe(tabela_final, hide_index=True, use_container_width=True)
                else:
                    with st.spinner(f"Simulando até a rodada {rodada_simulacao}..."):
                        tabela_simulada = simular_campeonato(rodada_simulacao, df_futuro, df_resultados, modelo, encoder, time_stats, colunas_modelo)
                    st.success(f"Tabela simulada até a rodada {rodada_simulacao}:")
                    st.dataframe(tabela_simulada, hide_index=True, use_container_width=True)

    # ----- Aba de Confronto Direto -----
    with tab_confronto:
        st.header("Confronto Direto", anchor=None)
        st.markdown("Analise o histórico de partidas entre dois clubes (com base nos dados carregados).")
        
        col1, col2 = st.columns(2)
        with col1:
            time1 = st.selectbox("Primeiro time:", lista_times, index=None, key="time1_h2h", placeholder="Escolha o primeiro time")
        with col2:
            time2 = st.selectbox("Segundo time:", lista_times, index=None, key="time2_h2h", placeholder="EscolGcolha o segundo time")

        if st.button("Analisar Confronto", use_container_width=False, type="primary", key="btn_confronto"):
            if time1 and time2:
                if time1 == time2:
                    st.warning("Por favor, escolha dois times diferentes.")
                else:
                    with st.spinner("Buscando histórico..."):
                        resumo, historico_df = gerar_confronto_direto(df_total, time1, time2)
                    
                    if resumo is None:
                        st.info(f"Não há jogos registrados entre {time1} e {time2} nos dados carregados.")
                    else:
                        st.subheader(f"Resumo: {time1} vs {time2}")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(f"Vitórias {time1}", resumo['vitorias'].get(time1, 0))
                        col2.metric("Empates", resumo.get('empates', 0))
                        col3.metric(f"Vitórias {time2}", resumo['vitorias'].get(time2, 0))
                        col4.metric("Total de Partidas", resumo.get('total_partidas', 0))

                        st.subheader("Histórico de Jogos")
                        st.dataframe(historico_df, hide_index=True, use_container_width=True)
            else:
                st.error("Selecione dois times para continuar.")
else:
    st.error("Não foi possível carregar os dados. Verifique a conexão ou tente novamente.")
    st.info("Se o erro persistir, pode ser um bloqueio temporário do servidor de dados ou falha na importação dos módulos.")