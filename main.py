import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os


logo = 'favicon.ico'
st.set_page_config(page_title='Detecção de Doenças no Algodão',
                   page_icon=logo, layout='wide',
                   initial_sidebar_state='expanded'
                   )

st.subheader('⚠️Detecção de Doenças no Algodão🌱', divider='rainbow')


HOME = os.getcwd()
print(HOME)

# Sidebar
st.sidebar.header('⚠️Detecção de Doenças no Algodão🌱')


image_path = "https://distanciamentosocial.streamlit.app/~/+/media/c0acdfa0cd0543f70281795b9a3038ff0664840d9956acc97fb71df9.png"
text = "Developed and Maintained by: Rian.Bispo"
rodape = st.sidebar.image(image_path, caption=text)

st.sidebar.markdown('''  
  [![Linkedin Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/rian-bispo/)](https://www.linkedin.com/in/rian-bispo/)
''')
###
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('''
## Objetivo do Projeto

O principal objetivo deste projeto é desenvolver um sistema de detecção simples e preciso. Esse sistema ajudará agricultores a identificar rapidamente problemas de saúde em suas plantações de algodão. A detecção precoce de doenças é crucial para a implementação de medidas preventivas e o controle efetivo, contribuindo para uma produção saudável e sustentável.''')
with col2:
    st.markdown('''
## Metodologia

Vamos utilizar técnicas avançadas de processamento de imagens e machine learning, especialmente Convolutional Neural Networks (CNN), para treinar um modelo capaz de reconhecer padrões associados a diferentes doenças em folhas de algodão. Este projeto concentrar-se-á em técnicas de classificação para a implementação e avaliação do modelo.''')
with col3:
    st.markdown('''
## Conjunto de Dados

Para atingir nossos objetivos, faremos uso de um conjunto de dados abrangente contendo imagens rotuladas de folhas de algodão saudáveis e afetadas por diversas doenças.
''')

# Certifique-se de substituir 'caminho_do_modelo.h5' pelo caminho real do seu arquivo.
caminho_do_modelo = r'C:/Users/rianb/PycharmProjects/Previsão de doenças do algodão/modelo_cnn.h5'

try:
    # Usar a função open para abrir o arquivo corretamente.
    with open(caminho_do_modelo, 'rb') as file:
        modelo_carregado = tf.keras.models.load_model(file)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name


# Caminho do vídeo de entrada
INPUT_PATH = st.file_uploader("Escolha uma Imagem", type='jpg')

if INPUT_PATH is not None:
    # Converta o objeto retornado por st.file_uploader para um caminho de arquivo
    img_path = save_uploaded_file(INPUT_PATH)

    # Carregando a imagem e redimensionando para o tamanho esperado (64x64)
    img = image.load_img(img_path, target_size=(64, 64))

    # Convertendo a imagem para um array numpy
    img_array = image.img_to_array(img)

    # Normalizando os valores dos pixels para o intervalo [0, 1]
    img_array = img_array / 255.0

    # Adicionando uma dimensão extra para representar o batch (pois o modelo espera um batch de imagens)
    img_array = np.expand_dims(img_array, axis=0)

    # Realizando a predição usando o modelo carregado
    prediction = modelo_carregado.predict(img_array)

    # Obtendo as classes previstas
    predicted_class_index = np.argmax(prediction)
    classes = ["diseased_leaf", "diseased_plant", "freash_leaf", "freash_plant"]
    predicted_class = classes[predicted_class_index]

    # Exibindo os resultados
    st.write("Resultado da predição:", prediction)
    st.write("Classe prevista:", predicted_class)
