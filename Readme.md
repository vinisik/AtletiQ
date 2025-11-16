# **AtletiQ**

Este é um aplicativo web desenvolvido em Python com Streamlit que utiliza técnicas de web scraping e machine learning para analisar e prever resultados de partidas do Campeonato Brasileiro de Futebol.

A aplicação busca dados atualizados, treina um modelo de regressão logística e apresenta diversas funcionalidades analíticas em uma interface amigável.

## **Funcionalidades**

* **Previsão de Jogo Específico:** Calcula as probabilidades de vitória, empate ou derrota para qualquer partida futura da temporada, incluindo a conversão para odds decimais.  
* **Simulação da Tabela:** Simula todos os jogos restantes até o final do campeonato e apresenta a provável tabela de classificação final.  
* **Confronto Direto:** Mostra o histórico completo de confrontos entre dois times, combinando uma base de dados histórica com os resultados da temporada atual.

## **Tecnologias Utilizadas**

* **Linguagem:** Python  
* **Interface Web:** Streamlit  
* **Manipulação de Dados:** Pandas & NumPy  
* **Machine Learning:** Scikit-learn  
* **Web Scraping:** Cloudscraper & Requests  
* **Dados Históricos:** Arquivo CSV local

## Executar o sreamlit
streamlit run app.py

### **Dados a Serem Armazenados no Banco de Dados:**

* **Resultados de Jogos Passados:** Todos os placares, datas e estatísticas de partidas já concluídas.  
* **Calendário de Jogos Futuros:** A lista de partidas a serem disputadas.  
* **Dados de Temporadas Anteriores:** Histórico completo de campeonatos passados para treinar modelos mais precisos.  
* **Confrontos Históricos (Head-to-Head):** A base de dados que hoje está no arquivo .csv será migrada para uma tabela no banco.  
* **Estatísticas de Jogadores:** O desempenho de todos os jogadores a cada temporada.

### **O Papel do Web Scraping:**

O web scraper continuará sendo essencial, mas seu papel mudará: ele será usado exclusivamente para **obter os dados da última rodada** (placares de jogos que acabaram de acontecer e estatísticas atualizadas dos jogadores) e alimentar o banco de dados. Ele não será mais necessário para construir o conjunto de dados do zero a cada execução do app.

## **Estrutura dos Arquivos**

* app.py: O arquivo principal que executa a interface web com Streamlit.  
* web\_scraper.py: Contém as funções para buscar dados de jogos e jogadores no site FBref.  
* feature\_engineering.py: Prepara os dados brutos, criando features para o modelo.  
* model\_trainer.py: Responsável por treinar o modelo de machine learning.  
* predictor.py: Contém a lógica para fazer previsões e simular a tabela.  
* analysis.py: Funções para as análises de confronto direto, por time e do campeonato.  
* historico\_confrontos.csv: Base de dados local com o histórico de confrontos.  
* requirements.txt: Lista de dependências do projeto.
