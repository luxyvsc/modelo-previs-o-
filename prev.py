# Desativar otimizações oneDNN e reduzir logs do TensorFlow antes de importá-lo.
# TF_ENABLE_ONEDNN_OPTS=0 desativa as mensagens sobre oneDNN.
# TF_CPP_MIN_LOG_LEVEL: 0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_jit(True)  # Ativa XLA JIT compiler
        print("✅ GPU detectada e otimizada!")
    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU NÃO DETECTADA - TensorFlow está usando CPU")
    
print(tf.__version__)
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

# Fonte dos dados
url = 'https://github.com/allanspadini/curso-tensorflow-proxima-palavra/raw/main/dados/train.zip'

# Agora ler o CSV a partir do arquivo local (pandas aceita .zip se contiver um único arquivo CSV)
df = pd.read_csv('train.zip', header=None, names=['ClassIndex','Título','Descrição'])

df['Texto'] = df['Título'] + ' ' + df['Descrição']

#tratamento de texto para conter menos dados


random.seed(42)
df_sample = df.sample(n = 1000)

corpus = df_sample['Texto'].tolist()

max_vocab_size = 20000
max_sequence_length = 50

vectorizer = TextVectorization(max_tokens=max_vocab_size, output_sequence_length=max_sequence_length, output_mode='int')

vectorizer.adapt(corpus)

vectorizer_model = tf.keras.Sequential([vectorizer])
vectorizer_model.save("vectorizer.keras")
print("✅ Vectorizer salvo com sucesso usando formato .keras")

tokenized_corpus = vectorizer(corpus)

input_sequences = []

for token_list in tokenized_corpus.numpy():
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
def prepare_sequences(sequences):
    """
    Prepara as sequências para o modelo, removendo zeros à direita, adicionando padding à esquerda, truncado sequências longas e removendo sequências repetidas.

    Args:
        sequences: Um array de sequências (listas ou arrays NumPy).

    Returns:
        Um array NumPy 2D com as sequências preparadas.
    """

    # Remover zeros à direita de cada sequência
    sequences_without_trailing_zeros = []
    for seq in sequences:
        last_nonzero_index = np.argmax(seq[::-1] != 0)
        if last_nonzero_index == 0 and seq[-1] == 0:
            sequences_without_trailing_zeros.append(np.array([0])) 
        else:
            sequences_without_trailing_zeros.append(seq[:-last_nonzero_index or None]) 

    # Remover sequências repetidas
    unique_sequences = []
    for seq in sequences_without_trailing_zeros:
        if seq.tolist() not in unique_sequences:  # Verifica se a sequência já está na lista
            unique_sequences.append(seq.tolist())  # Adiciona à lista se for única

    # Encontrar o comprimento máximo das sequências sem zeros à direita
    max_sequence_len = max(len(seq) for seq in unique_sequences)

    # Adicionar padding à esquerda para garantir o mesmo comprimento
    padded_sequences = pad_sequences(unique_sequences, maxlen=max_sequence_len, padding='pre', truncating='post')

    return padded_sequences

input_sequences_prepared = prepare_sequences(input_sequences)

X = input_sequences_prepared[:, :-1]
y = input_sequences_prepared[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=max_vocab_size)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=max_vocab_size,
        output_dim=128,
        mask_zero=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48)),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(max_vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])



# Treinar o modelo
with tf.device('/GPU:0'):
    history = model.fit(X, y, epochs=100, verbose=1, batch_size=512)
    
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.show()

def predict_next_word(model, vectorizer, text, max_sequence_length, top_k=3):
    """
    Prediz a próxima palavra para um dado texto de entrada usando o modelo treinado e o vetor de texto.

    Args:
        model: O modelo treinado.
        vectorizer: O vetor de texto usado para tokenização.
        text: A string de entrada para a qual prever a próxima palavra.
        max_sequence_length: O comprimento máximo da sequência usada no treinamento.

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

text = "The Fbi is warning consumers against using public phone charging stations in order to"

predict_next_word(model, vectorizer, text, max_sequence_length, top_k=3)

model.save('next_word_model.keras')
print("✅ Modelo salvo com sucesso usando formato .keras")