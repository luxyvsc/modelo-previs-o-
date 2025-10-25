import streamlit as st
import tensorflow as tf
import numpy as np
import pickle 
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model():
    model = tf.keras.models.load_model('next_word_model.keras')
    vectorizer_model = tf.keras.models.load_model('vectorizer.keras')
    vectorizer = vectorizer_model.layers[0]  # o vectorizer está como camada 0
    return model, vectorizer

def predict_next_word(model, vectorizer, text, max_sequence_length, top_k=1):
    """
    Prediz a próxima palavra para um dado texto de entrada usando o modelo treinado e o vetor de texto.

    Args:
        model: O modelo treinado.
        vectorizer: O vetor de texto usado para tokenização.
        text: A string de entrada para a qual prever a próxima palavra.
        max_sequence_length: O comprimento máximo da sequência usada no treinamento.
        top_k: O número de palavras principais a serem retornadas.
        
    Returns:
        A palavra prevista como a próxima palavra.
    """
    tokenized_text = vectorizer([text])
    tokenized_text = np.squeeze(tokenized_text)
    
    padded_text = pad_sequences([tokenized_text], maxlen=max_sequence_length-1, padding='pre')
    
    predict_probs = model.predict(padded_text, verbose=0)[0]
    
    tok_k_indices = np.argsort(predict_probs)[-top_k:][::-1]
    predicted_words = [vectorizer.get_vocabulary()[i] for i in tok_k_indices]
    return predicted_words

def main():
    max_sequence_len = 50

    model, vectorizer = load_model()

    st.title('Predição de Próxima Palavra')
    
    input_text = st.text_input('Digite uma frase:')
    
    if st.button('Prever Próxima Palavra'):
        if input_text:
            try:
                predicted_words = predict_next_word(model, vectorizer, input_text, max_sequence_len, top_k=3)
                st.info(f'Próxima palavra prevista: {predicted_words[0]}')
                for word in predicted_words:
                    st.success(word)
            except Exception as e:
                st.error(f'Ocorreu um erro durante a predição: {e}')
        else:
            st.warning('Por favor, insira uma frase.')
    
if __name__ == '__main__':
    main()