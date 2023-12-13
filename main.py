import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing import image

# Caminho da imagem que deseja fazer a predição
img_path = '/content/PrevisaodeDoencasnoAlgodao/Cotton Disease/test/diseased cotton leaf/dis_leaf (124).jpg'

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
print("Resultado da predição:", prediction)
print("Classe prevista:", predicted_class)
