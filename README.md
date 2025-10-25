# Prevendo Palavras

Este projeto utiliza um modelo de aprendizado de máquina para prever a próxima palavra em uma sequência de texto. Ele foi desenvolvido em Python e utiliza modelos treinados salvos nos arquivos `.keras`.

## Estrutura do Projeto

- `app.py`: Script principal para executar a aplicação.
- `prev.py`: Script com funções auxiliares ou lógicas relacionadas à previsão de palavras.
- `next_word_model.keras`: Modelo treinado para previsão da próxima palavra.
- `vectorizer.keras`: Vetorizador treinado para processar o texto de entrada.
- `requirements.txt`: Lista de dependências do projeto.

## Como Executar

1. **Instale as dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Execute o script principal:**
   ```powershell
   streamlit run app.py
   ```

## Requisitos

- Python 3.8+
- Pacotes listados em `requirements.txt`

## Funcionamento

O projeto carrega um modelo treinado e um vetorizador para processar o texto de entrada e prever a próxima palavra provável na sequência.

## Licença

Este projeto é de uso acadêmico/educacional. Modifique conforme necessário para seu uso.
