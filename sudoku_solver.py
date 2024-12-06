import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import tensorflow as tf
from imutils import contours
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
            rescale=1./255
        )

        train_data = datagen.flow_from_directory(
            directory=f"{base_path}/train_data",
            target_size=(70, 70),
            color_mode='grayscale',
            batch_size=128,
            class_mode='sparse'
        )

        val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
            directory=f"{base_path}/validation_data",
            target_size=(70, 70),
            color_mode='grayscale',
            batch_size=128,
            class_mode='sparse'
        )

        # define a arquitetura do modelo
        modelo = Sequential([ # sequential é um tipo de modelo onde as camadas são empilhadas sequencialmente. ou seja, cada camada recebe a saída da camada anterior como entrada. maneira fácil de criar redes neurais
            Conv2D(32, (3, 3), activation='relu', input_shape=(70, 70, 1)), #Conv2D: é uma camada convolucional 2D. Ajuda a detectar padrões locais (como bordas, texturas, etc.) em imagens. / 32: é o número de filtros (ou kernels). filtros são pequenos arrays que percorrem a imagem para extrair características.
            MaxPooling2D((2, 2)), # (3, 3) do conv2d: O tamanho do filtro é 3x3, ou seja, ele examina uma área de 3x3 pixels da imagem de entrada. / activation='relu': a função de ativação relu (rectified linear unit) é aplicada, que transforma valores negativos em zero e mantém os valores positivos.
            Conv2D(64, (3, 3), activation='relu'), # input_shape=(70, 70, 1): Define o tamanho da entrada da imagem. Nesse caso, é uma imagem de 70x70 pixels com 1 canal (imagens em escala de cinza).
            MaxPooling2D((2, 2)), # MaxPooling2D: é uma camada de pooling (subamostragem). o pooling reduz a dimensionalidade da imagem ao pegar o valor máximo de uma região específica. isso ajuda a reduzir a complexidade e a controlar o overfitting. / (2, 2): o tamanho da janela de pooling é 2x2, ou seja, ele vai pegar a maior valor em cada bloco de 2x2 pixels da imagem.
            Conv2D(128, (3, 3), activation='relu'), #128: aumento no número de filtros permite que a rede aprenda representações mais complexas da imagem.
            MaxPooling2D((2, 2)),
            Flatten(), #essa camada "achata" a saída das camadas anteriores (que têm forma tridimensional) em um vetor unidimensional. ou seja, ele transforma a matriz resultante das camadas convolucionais e de pooling em um vetor de uma única linha que pode ser alimentado em uma rede neural densa.
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)), #dense: é uma camada densa totalmente conectada, onde todos os neurônios da camada anterior estão conectados a cada neurônio dessa camada. / 128: O número de neurônios na camada densa.
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)), #kernel_regularizer=l2(0.01): a regularização L2 é aplicada a essa camada, ajudando a evitar overfitting. a regularização L2 penaliza grandes valores de pesos, ajudando o modelo a não se ajustar excessivamente aos dados de treinamento.
            Dense(10, activation='softmax') # Dense(10): A camada final, com 10 neurônios, geralmente usada para problemas de classificação com 10 classes (por exemplo, no caso estamos lidando com 10 dígitos, como 0-9 em reconhecimento de números).
        ]) #activation='softmax': A função Softmax é aplicada, transformando as saídas em probabilidades, ou seja, a saída de cada neurônio será entre 0 e 1, e a soma de todas as saídas será 1.

        modelo.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-6
        )

        # treinamento
        modelo.fit(
            train_data,
            epochs=50,
            validation_data=val_data,
            callbacks=[early_stopping, reduce_lr]
        )

        # salva modelo treinado
        modelo.save("modelo_sudoku.keras")
        print("Modelo treinado e salvo com sucesso!")

    return modelo


def monta_sudoku(linhas_sudoku, modelo, img):
    sudoku_matriz = [[0 for _ in range(9)] for _ in range(9)]
    
    for i, row in enumerate(linhas_sudoku):
        for j, c in enumerate(row): 
            # cria uma máscara para isolar a célula
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            result = cv2.bitwise_and(img, mask)
            cv2.imshow('result', result)
            cv2.waitKey(600)
            # extrai a célula (bounding box do contorno)
            x, y, w, h = cv2.boundingRect(c)
            numero = result[y:y + h, x:x + w]
            
            # verifica se a imagem tem mais de 1 canal (BGR)
            if len(numero.shape) == 3:  
                numero = cv2.cvtColor(numero, cv2.COLOR_BGR2GRAY)  # converte para escala de cinza
            
            # pré-processamento: redimensionar para 70x70 e binarizar
            numero = cv2.resize(numero, (70, 70), interpolation=cv2.INTER_AREA)
            numero = cv2.threshold(numero, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # verifica se a célula está vazia (contando pixels pretos)
            if np.sum(numero == 0) < 30:  # se há poucos pixels pretos, considere vazio
                valor = 0
            else:
                numero = numero.reshape(1, 70, 70, -1) / 255.0  # normalizar entre 0 e 1
                # classifica o número utilizando o modelo treinado
                predicao = modelo.predict(numero)
                valor = np.argmax(predicao)
                # verifica confiança da previsão
                if predicao.max() < 0.7:
                    valor = 0

            sudoku_matriz[i][j] = valor

            # mostra cada célula isolada e seu valor detectado
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
                for numero in range(1,10): # itera sobre todos os números possíveis (1 a 9) para tentar preencher a célula.
                    if verifica_numero(matriz,linha,coluna,numero): # verifica se o número atual pode ser colocado na célula, sem violar as regras do Sudoku.
                        matriz[linha][coluna] = numero #se o número for válido, ele é temporariamente colocado na célula.
                        if resolve_sudoku(matriz): #a função é chamada recursivamente para tentar resolver o restante do tabuleiro.
                            return True
                        matriz[linha][coluna] = 0 # se a solução não for possível com o número atual, a célula é redefinida para 0 e o algoritmo tenta o próximo número.(backtracking)
                return False #se nenhum número de 1 a 9 for válido, retorna false, indicando que o Sudoku não pode ser resolvido com a configuração atual.
    return True #quando todas as células estão preenchidas corretamente, retorna true.

def escrever_resultado_imagem(matriz,linhas_sudoku,img):
    for i, linha in enumerate(linhas_sudoku): #percorre cada linha identificada na imagem, indexando com i
        for j, coluna in enumerate(linha): #para cada célula da linha, obtém a posição da coluna correspondente e o índice j.
            x,y,w,h = cv2.boundingRect(coluna) #usando o OpenCV, calcula as coordenadas (x, y) e o tamanho (largura w e altura h) da célula detectada.
            if matriz[i][j] != 0: #apenas escreve na imagem os números preenchidos (ou seja, os diferentes de 0).
                numero = str(matriz[i][j]) #converte o número da matriz em uma string para poder escrevê-lo na imagem.
                fonte = cv2.FONT_HERSHEY_SIMPLEX 
                fonte_tamanho = 0.8 
                grossura = 2 
                tamanho_numero = cv2.getTextSize(numero,fonte,fonte_tamanho,grossura)[0] #calcula o deslocamento necessário para centralizar o número dentro da célula com base no tamanho do texto e nas dimensões da célula.
                numero_x = x + (w - tamanho_numero[0]) // 2
                numero_y = y + (h + tamanho_numero[1]) // 2
                cv2.putText(img,numero,(numero_x,numero_y), fonte, fonte_tamanho, (255,0,0), grossura) #escreve o número na posição centralizada, com a cor azul (255,0, 0) e a grossura definida.

def main():
    #coloque o caminha para sua imagem aqui, ela tem que ser uma grade de sudoku 640x640.
    img_path = "C:\\Users\\nicol\\OneDrive\\Desktop\\s5.png"

    # passo 1: carregar e binarizar a imagem
    img, img_binaria = carregar_alterar_img(img_path)
    cv2.imshow('Imagem Binarizada', img_binaria)

    # passo 2: filtrar e corrigir linhas
    img_binaria = detecta_contornos_arruma_linhas(img_binaria)

    # passo 3: detectar linhas da grade
    linhas_sudoku = ordena_grade(img_binaria)

    # passo 4: treinar ou carregar o modelo
    modelo = treina_ia_carrega_ia()

    # passo 5: construir matriz do Sudoku
    sudoku_matriz = monta_sudoku(linhas_sudoku, modelo, img)

    # passo 6: resolver o jogo
    print("Matriz inicial detectada:")
    for linha in sudoku_matriz:
        print(linha)

    if resolve_sudoku(sudoku_matriz):
        print("Jogo resolvido:")
        for linha in sudoku_matriz:
            print(linha)
    else:
        print("Não foi possível resolver o Sudoku.")

    # passo 7: escrever o resultado na imagem
    escrever_resultado_imagem(sudoku_matriz, linhas_sudoku, img)
    cv2.imshow("SSS", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
