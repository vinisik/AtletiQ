import pandas as pd
import cloudscraper
from io import StringIO
from typing import Optional


def buscar_dados_brasileirao(ano: str) -> Optional[pd.DataFrame]:
    """
    Busca os resultados e jogos futuros de uma temporada do Brasileirão no FBref,
    utilizando cloudscraper para contornar proteções anti-bot (Cloudflare).
    """
    print(f"Buscando dados da temporada {ano}...")

    url = f"https://fbref.com/en/comps/24/schedule/{ano}-Serie-A-Scores-and-Fixtures"

    # Criar scraper com cabeçalho de navegador
    scraper = cloudscraper.create_scraper(
        browser={'custom': 'chrome', 'platform': 'windows', 'mobile': False}
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }

    try:
        # Fazer requisição simulando um navegador real
        response = scraper.get(url, headers=headers, timeout=20)
        response.raise_for_status()  # levanta erro se status != 200

        # Ler tabelas do HTML
        tabelas = pd.read_html(StringIO(response.text), match="Scores & Fixtures")

        if not tabelas:
            print(f"AVISO: Nenhuma tabela com 'Scores & Fixtures' foi encontrada para o ano {ano}.")
            return None

        df = tabelas[0]

        # Limpeza básica
        df = df[df['Wk'].notna()]
        df = df[pd.to_numeric(df['Wk'], errors='coerce').notna()]

        # Renomear colunas
        df = df.rename(columns={
            'Wk': 'Rodada',
            'Home': 'HomeTeam',
            'Away': 'AwayTeam',
            'Score': 'Result'
        })

        # Separar gols
        if 'Result' in df.columns:
            gols = df['Result'].str.split('–', expand=True)
            df['FTHG'] = pd.to_numeric(gols[0], errors='coerce')
            df['FTAG'] = pd.to_numeric(gols[1], errors='coerce')

        # Colunas finais
        colunas_finais = ['Rodada', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        df = df[[c for c in colunas_finais if c in df.columns]]

        print(f"✅ Dados de {ano} carregados com sucesso. Total de {len(df)} partidas.")
        return df

    except Exception as e:
        print(f"❌ ERRO ao buscar ou processar dados de {ano}: {e}")
        return None
