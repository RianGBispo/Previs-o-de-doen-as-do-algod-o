import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

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
st.sidebar.markdown('---')
st.sidebar.subheader('️Como Usar')
texto = '''
1. Faça upload da Imagem que deseja Classificar
2. Agurde...
3. Visualize o resultado Gerado
'''
st.sidebar.markdown(texto)

status = st.sidebar.empty()

image_path = "https://distanciamentosocial.streamlit.app/~/+/media/c0acdfa0cd0543f70281795b9a3038ff0664840d9956acc97fb71df9.png"
text = "Developed and Maintained by: Rian.Bispo"
rodape = st.sidebar.image(image_path, caption=text, width=300)

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

# Recreate the exact same model, including its weights and the optimizer
modelo_carregado = tf.keras.models.load_model('modelo_cnn.h5')


def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name


def visualize_image_and_prediction(model, image_path, class_names):
    # Dicionário de tradução
    translation_dict = {
        'diseased_leaf': 'folha doente',
        'diseased_plant': 'planta doente',
        'fresh_leaf': 'folha saudável',
        'fresh_plant': 'planta saudável'
    }

    # Carregando a imagem
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image_array = image.img_to_array(test_image)
    test_image_array = test_image_array / 255.0  # Normalizando os valores dos pixels para o intervalo [0, 1]

    # Expandindo as dimensões para a previsão
    test_image_array = np.expand_dims(test_image_array, axis=0)

    # Realizando a previsão
    result = model.predict(test_image_array)

    # Mapeando os resultados para as classes
    predicted_class_index = np.argmax(result)
    predicted_class_english = class_names[predicted_class_index]
    predicted_class_portuguese = translation_dict.get(predicted_class_english, predicted_class_english)

    cl1, cl2 = st.columns(2)
    with cl1:
        # Exibindo os resultados
        # st.write("Resultado da predição:", predicted_class_index)
        st.image(image_path, caption=("Classe prevista: " + predicted_class_portuguese), width=500)
    with cl2:
        if predicted_class_index == 0:
            st.image('data/folhadoente.png', width=500, caption='Folha Doente')
        elif predicted_class_index == 1:
            st.image('data/plantadoente.png', width=500, caption='Planta Doente')
        elif predicted_class_index == 2:
            st.image('data/folha.png', width=500, caption='Folha Fresca')
        else:
            st.image('data/planta-crescente.png', width=500, caption='Planta Fresca')


# Caminho da imagem de entrada
INPUT_PATH = st.file_uploader("Escolha uma Imagem", type='jpg')

if INPUT_PATH is not None:
    try:
        # Atualize o status para indicar que a imagem está sendo carregada
        with st.spinner('Carregando a Imagem...'):
            time.sleep(1)

        # Carregue a imagem e realize a previsão
        image_path = INPUT_PATH
        class_names = ["diseased_leaf", "diseased_plant", "fresh_leaf", "fresh_plant"]
        visualize_image_and_prediction(modelo_carregado, image_path, class_names)

        # Atualize o status para indicar que a previsão foi concluída com sucesso
        status.success('Pronto!!!', icon="✅")

    except Exception as e:
        # Em caso de erro, atualize o status para indicar o problema
        status.error(f"Erro: {str(e)}", icon="❌")
