import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2

import tensorflow as tf

from imutils import contours

import numpy as np

from tensorflow.keras.models import load_model

import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from tensorflow.image import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets import mnist

from tensorflow.keras.callbacks import ReduceLROnPlateau



# carrega a imagem, deixa ela cinza, após isso deixa ela preta e branca

def carregar_alterar_img(img_path):

    img = cv2.imread(img_path)

    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_binaria = cv2.adaptiveThreshold(img_cinza,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,15)

    return img, img_binaria



# filtra os números e os quadrados da grade isolando-os

def detecta_contornos_arruma_linhas(img_binaria):

    contornos = cv2.findContours(img_binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # detecta contornos na imagem // cv2.CHAIN_APPROX_SIMPLE --> compacta os contornos // contornos = armazena os contornos

    contornos = contornos[0] if len(contornos) == 2 else contornos[1]

    

    for c in contornos:

        area = cv2.contourArea(c) # cv2.contourArea(c) --> calcula a area de cada contorno

        if area < 1000: # se a area do contorno for < 1000, ela é preenchida com preto para ser eliminada

            cv2.drawContours(img_binaria, [c], -1, (0,0,0), -1)



    # arruma as linhas horizontais e verticais para garantir que a grade seja um quadrado // cv2.morphologyEx --> fecha as lacunas verticais e verticais

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))

    img_binaria = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))

    img_binaria = cv2.morphologyEx(img_binaria, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)



    return img_binaria



def ordena_grade(img_binaria):

    invert = 255 - img_binaria

    contornos = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = contornos[0] if len(contornos) == 2 else contornos[1]

    (contornos, _) = contours.sort_contours(contornos, method="top-to-bottom") # ordena os contornos de cima para baixo, para separar as linhas da grade



    linhas_sudoku = [] # lista para armazenar as linhas completas do Sudoku

    row = [] # lista temporária para armazenar os contornos de uma linha

    for (i, c) in enumerate(contornos, 1): # itera pelos contornos detectados, numerando-os

        area = cv2.contourArea(c)

        if area < 50000:

            row.append(c)

            if i % 9 == 0:  # verifica se já temos 9 células/quadrados na linha

                (contornos, _) = contours.sort_contours(row, method="left-to-right")

                linhas_sudoku.append(contornos) # adiciona a linha completa na lista 

                row = []

    return linhas_sudoku



def treina_ia_carrega_ia():

    try:

        modelo = load_model("modelo_sudoku.keras")

        print("Modelo pré-treinado carregado com sucesso!")

    except (FileNotFoundError, ValueError):

        print("Treinando um novo modelo com dataset local...")



        # Caminho base do dataset

        base_path = './data/data_v2'



        # Geradores de dados para treino e validação

        datagen = ImageDataGenerator(

            rescale=1./255,

            width_shift_range=0.1,

            height_shift_range=0.1,

            shear_range=0.1,

            zoom_range=0.1

        )



        train_data = datagen.flow_from_directory(

            directory=f"{base_path}/train_data",

            target_size=(70, 70),

            color_mode='grayscale',

            batch_size=64,

            class_mode='sparse'

        )



        val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(

            directory=f"{base_path}/validation_data",

            target_size=(70, 70),

            color_mode='grayscale',

            batch_size=64,

            class_mode='sparse'

        )



        # Definir a arquitetura do modelo

        modelo = Sequential([

            Conv2D(32, (3, 3), activation='relu', input_shape=(70, 70, 1)),

            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),

            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),

            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),

            Dropout(0.5),

            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),

            Dropout(0.3),

            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),

            Dropout(0.2),

            Dense(10, activation='softmax')

        ])



        modelo.compile(

            optimizer=Adam(learning_rate=0.0001),

            loss='sparse_categorical_crossentropy',

            metrics=['accuracy']

        )



        # Callbacks

        early_stopping = EarlyStopping(

            monitor='val_loss',

            patience=15,

            restore_best_weights=True

        )

        reduce_lr = ReduceLROnPlateau(

            monitor='val_loss',

            factor=0.3,

            patience=10,

            min_lr=1e-6

        )



        # Treinamento

        modelo.fit(

            train_data,

            epochs=300,

            validation_data=val_data,

            callbacks=[early_stopping, reduce_lr]

        )



        # Salvar modelo treinado

        modelo.save("modelo_sudoku.keras")

        print("Modelo treinado e salvo com sucesso!")



    return modelo





def monta_sudoku(linhas_sudoku, modelo, img):

    sudoku_matriz = [[0 for _ in range(9)] for _ in range(9)]

    

    for i, row in enumerate(linhas_sudoku):

        for j, c in enumerate(row): 

            # Cria uma máscara para isolar a célula

            mask = np.zeros(img.shape, dtype=np.uint8)

            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

            result = cv2.bitwise_and(img, mask)

            cv2.imshow('result', result)

            cv2.waitKey(600)

            # Extrai a célula (bounding box do contorno)

            x, y, w, h = cv2.boundingRect(c)

            numero = result[y:y + h, x:x + w]

            

            # Verifica se a imagem tem mais de 1 canal (BGR)

            if len(numero.shape) == 3:  

                numero = cv2.cvtColor(numero, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza

            

            # Pré-processamento: Redimensionar para 70x70 e binarizar

            numero = cv2.resize(numero, (70, 70), interpolation=cv2.INTER_AREA)

            numero = cv2.threshold(numero, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            numero = 255 - numero  # Inverte para combinar com o padrão EMNIST

            

            # Verifica se a célula está vazia (contando pixels pretos)

            if np.sum(numero == 0) < 30:  # Se há poucos pixels pretos, considere vazio

                valor = 0

            else:

                numero = numero.reshape(-1, 70, 70, 1) / 255.0  # Normalizar entre 0 e 1

                # Classifica o número utilizando o modelo treinado

                predicao = modelo.predict(numero)

                valor = np.argmax(predicao)

                # Verifica confiança da previsão

                if predicao.max() < 0.8:

                    valor = 0



            sudoku_matriz[i][j] = valor



            # Para depuração: mostra cada célula isolada e seu valor detectado

            print(f"Valor detectado na célula [{i}][{j}]: {valor}")

    

    return sudoku_matriz



def resolve_sudoku(matriz):

    def verifica_numero(matriz, linha, coluna, numero):

        for i in range(9):

            if matriz[linha][i] == numero or matriz[i][coluna] == numero:

                return False

            if matriz[linha//3*3 + i//3][coluna //3*3 + i%3] == numero:

                return False

        return True

    

    for linha in range(9):

        for coluna in range(9):

            if matriz[linha][coluna] == 0:

                for numero in range(1,10):

                    if verifica_numero(matriz,linha,coluna,numero):

                        matriz[linha][coluna] = numero

                        if resolve_sudoku(matriz):

                            return True

                        matriz[linha][coluna] = 0

                return False

    return True



def escrever_resultado_imagem(matriz,linhas_sudoku,img):

    for i, linha in enumerate(linhas_sudoku):

        for j, coluna in enumerate(linha):

            x,y,w,h = cv2.boundingRect(coluna)

            if matriz[i][j] != 0:

                numero = str(matriz[i][j])

                fonte = cv2.FONT_HERSHEY_SIMPLEX

                fonte_tamanho = 0.8

                grossura = 2

                tamanho_numero = cv2.getTextSize(numero,fonte,fonte_tamanho,grossura)[0]

                numero_x = x + (w - tamanho_numero[0]) // 2

                numero_y = y + (h + tamanho_numero[1]) // 2

                cv2.putText(img,numero,(numero_x,numero_y), fonte, fonte_tamanho, (0,255,0), grossura)



def main():

    img_path = "D:\\downloads\\sudoku_ai\\s3.webp"



    # Passo 1: Carregar e binarizar a imagem

    img, img_binaria = carregar_alterar_img(img_path)

    cv2.imshow('Imagem Binarizada', img_binaria)



    # Passo 2: Filtrar e corrigir linhas

    img_binaria = detecta_contornos_arruma_linhas(img_binaria)



    # Passo 3: Detectar linhas da grade

    linhas_sudoku = ordena_grade(img_binaria)



    # Passo 4: Treinar ou carregar o modelo

    modelo = treina_ia_carrega_ia()



    # Passo 5: Construir matriz do Sudoku

    sudoku_matriz = monta_sudoku(linhas_sudoku, modelo, img)



    # Passo 6: Resolver o jogo

    print("Matriz inicial detectada:")

    for linha in sudoku_matriz:

        print(linha)



    if resolve_sudoku(sudoku_matriz):

        print("Jogo resolvido:")

        for linha in sudoku_matriz:

            print(linha)

    else:

        print("Não foi possível resolver o Sudoku.")



    # Passo 7: Escrever o resultado na imagem

    escrever_resultado_imagem(sudoku_matriz, linhas_sudoku, img)

    cv2.imshow("SSS", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()





if __name__ == "__main__":

    main()


