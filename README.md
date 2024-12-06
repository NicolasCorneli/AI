# 🧩 IA para Resolução de Sudoku

## ✨ Justificativa da Abordagem de IA Escolhida

A escolha de uma rede neural conexionista foi motivada pela vontade de aprender e explorar o uso de redes neurais. A rede neural é ideal para essa tarefa porque pode aprender a partir de exemplos, ajustando-se aos padrões complexos presentes nas imagens dos números.

## 🎲 Descrição e Origem do Dataset

O dataset utilizado foi selecionado do site Kaggle e adaptado para o objetivo de identificar números digitais no Sudoku.

## 🧠 Processos de Treinamento/Ajuste e Teste

O treinamento foi realizado utilizando um modelo de rede neural convolucional (CNN) com três camadas convolucionais, camadas de pooling e camadas densas para classificar os números. O modelo foi compilado com o otimizador Adam, a função de perda `sparse_categorical_crossentropy` e a métrica de precisão (`accuracy`).

### Parâmetros do Modelo
- Camadas convolucionais: 32, 64 e 128 filtros, com tamanhos de kernel (3, 3).
- Camadas de pooling: MaxPooling com tamanho (2, 2).
- Camadas densas: 128 unidades, com regularização L2 (`kernel_regularizer=l2(0.01)`), para evitar overfitting.
- Taxa de aprendizado: 0.0001.

### Métricas de Precisão
Durante o treinamento, a precisão foi monitorada utilizando os dados de validação. O modelo foi ajustado utilizando técnicas de `early stopping` e `reduce_lr_on_plateau` para evitar overfitting e garantir que o modelo continuasse a aprender de forma eficiente.

## 🔎 Análise Crítica dos Resultados

### Dificuldades Encontradas
Uma das maiores dificuldades enfrentadas durante o desenvolvimento foi lidar com o pré-processamento de imagens. Adaptar, redirecionar, e iterar pelas imagens do Sudoku de forma eficaz foi um desafio significativo. As imagens precisaram ser binarizadas e redimensionadas, e isso exigiu bastante experimentação para encontrar os melhores parâmetros.

### Fine-Tuning
O fine-tuning do modelo foi particularmente desafiador, pois era necessário encontrar o equilíbrio entre a aprendizagem eficiente e a prevenção do overfitting. Para isso, foi ajustada a arquitetura da rede, bem como os hiperparâmetros, como a taxa de aprendizado e a regularização L2. A escolha de uma taxa de aprendizado baixa (0.0001) e a aplicação da regularização L2 ajudaram a garantir que o modelo não fosse excessivamente complexo e, assim, evitou-se que o modelo ficasse “desprovido de inteligência” ou apresentasse overfitting.


## 📜 Descrição do Projeto

Este projeto visa a construção de uma Inteligência Artificial (IA) do zero, com a aplicação de técnicas de redes neurais para resolver o jogo de Sudoku a partir de imagens. A IA foi desenvolvida com o uso da biblioteca TensorFlow e Keras, e o modelo treinado utiliza uma arquitetura de rede neural convolucional (CNN) para classificar os números nas células de uma grade de Sudoku.

## 🤖 O que é uma CNN ?

Uma Rede Neural Convolucional (ConvNet / Convolutional Neural Network / CNN) é um algoritmo de Aprendizado Profundo que pode captar uma imagem de entrada, atribuir importância (pesos e vieses que podem ser aprendidos) a vários aspectos / objetos da imagem e ser capaz de diferenciar um do outro.

## 🎯 Objetivo do Trabalho

O objetivo deste trabalho é aplicar o conceito de **Conexionismo**, com o uso de redes neurais para a resolução de um problema clássico de raciocínio lógico: o Sudoku. A IA foi desenvolvida para ler imagens do jogo e identificar os números.

## 🛠️ Etapas do Desenvolvimento

1. **Escolha do paradigma de IA**:
   - A abordagem escolhida foi o **Conexionismo**, com a utilização de redes neurais para o reconhecimento dos números na grade do Sudoku.

2. **Criação/seleção de Dataset**:
   - Foi utilizado um dataset de imagens contendo os números das células da grade do Sudoku, onde essas imagens das células com e sem números(0) foram rotulados para treinamento do modelo de IA.

3. **Construção e Ajuste do Modelo**:
   - O modelo foi desenvolvido utilizando a arquitetura **Convolutional Neural Network (CNN)**, com camadas de convolução, pooling, e camadas densas para a classificação dos números.

4. **Teste e Validação**:
   - Foi feita a divisão do dataset para treinamento e teste para validar o modelo. Além disso, durante o treinamento, a precisão foi monitorada continuamente, e ajustes foram feitos conforme necessário para aprimorar a performance do modelo.

5. **Reajuste e Fine-Tuning**:
   - O modelo passou por um processo de fine-tuning, onde hiperparâmetros foram ajustados e a arquitetura foi refinada para otimizar a aprendizagem dos números. Isso permitiu que o modelo identificasse os números de forma mais eficiente e precisa, melhorando seu       
     desempenho na resolução do Sudoku.

## 📋 Requisitos Técnicos

- ### **Bibliotecas**:
  - ### TensorFlow
  - O TensorFlow é uma biblioteca de aprendizado de máquina usada para criar, treinar e testar modelos de IA. No código, ele é utilizado para construir e treinar a rede neural convolucional (CNN) que faz o reconhecimento dos números nas células do         Sudoku. O         TensorFlow gerencia a criação do modelo, o fluxo de dados, o cálculo de gradientes e a otimização dos pesos durante o treinamento.

  - ### Keras
  - O Keras é uma API de alto nível que facilita a construção e treinamento de modelos de aprendizado profundo, como redes neurais. Ele é usado em conjunto com o TensorFlow para criar e treinar a arquitetura da rede neural (CNN) que identifica os          números do        Sudoku. Embora o TensorFlow forneça a base para as operações de baixo nível, o Keras oferece uma interface simplificada para construir, compilar e treinar o modelo.

  - ### OpenCV
  - O OpenCV (Open Source Computer Vision Library) é uma biblioteca para processamento de imagens. No código, ela é utilizada para manipular e processar as imagens do Sudoku, como carregá-las, converter para escala de cinza, binarizar, detectar            contornos e       identificar as células da grade. O OpenCV facilita a extração de informações visuais da imagem antes de passar para a rede neural.
    
  - ### NumPy
  - O NumPy é uma biblioteca fundamental para cálculos numéricos em Python, oferecendo suporte a arrays multidimensionais e funções matemáticas rápidas. No código, ele é usado para trabalhar com arrays de dados, como as matrizes que representam as         grades de         Sudoku. O NumPy também é essencial para realizar operações eficientes com dados numéricos, como matrizes e vetores, que são usados no treinamento da rede neural.

  - ### Imutils
  - O Imutils é uma biblioteca auxiliar para simplificar várias operações de processamento de imagens, como redimensionamento, rotação e transformação. No código, ele é utilizado para os contornos da grade que foram detectados.

- ### **Ferramentas**:
  - Python 3.x
  - TensorFlow e Keras para treinamento e construção do modelo de IA.
  - OpenCV para processamento de imagens.

## ⚙️ Como Usar

### Passos para Rodar o Projeto

1. **Instalação das Dependências**:
   - Clone o repositório:
     ```bash
     git clone https://github.com/NicolasCorneli/AI
     cd AI
     ```
   - Instale as dependências:
     ```bash
     pip install -r requirements.txt
     ```

2. **Treinamento do Modelo**:
   - Caso não tenha um modelo pré-treinado, o sistema irá treinar um novo modelo com o dataset local. Para treinar, execute o script:
     ```bash
     python sudoku_ai.py
     ```

3. **Execução do Script de Resolução**:
   - Para rodar o script e resolver uma imagem de Sudoku, basta fornecer o caminho da imagem:
     ```bash
     python sudoku_ai.py --img_path caminho/para/imagem/sudoku.png
     ```

4. **Modelo Pré-Treinado**:
   - O modelo treinado será salvo como `modelo_sudoku.keras`. Caso já tenha um modelo, ele será carregado automaticamente.

## 🔧 Funções Principais

1. **Carregar e Processar Imagens**:
   - O código usa a função `carregar_alterar_img` para carregar a imagem e convertê-la para escala de cinza e binarizá-la.

2. **Detecção de Contornos e Linhas**:
   - A função `detecta_contornos_arruma_linhas` detecta os contornos na imagem para identificar as células do Sudoku.

3. **Treinamento e Carregamento do Modelo**:
   - A função `treina_ia_carrega_ia` treina um modelo de rede neural utilizando o dataset fornecido ou carrega um modelo pré-existente.

4. **Construção da Matriz do Sudoku**:
   - A função `monta_sudoku` usa a rede neural para identificar e classificar os números em cada célula da grade de Sudoku.

5. **Resolução do Sudoku**:
   - A função `resolve_sudoku` usa o algoritmo de backtracking para resolver a matriz do Sudoku preenchida pela IA.

6. **Escrita do Resultado na Imagem**:
   - A função `escrever_resultado_imagem` escreve os números resolvidos de volta na imagem original.

## 🎯 Resultados Esperados

Ao executar o script, a IA deve ser capaz de identificar e classificar os números em cada célula da grade de Sudoku presente em uma imagem, para que assim o resultado do jogo seja calculado e mostrado diretamente sobre a imagem original.

## 📝 Conclusões

Este projeto demonstra como é possível utilizar redes neurais convolucionais para resolver problemas lógicos simples, como o Sudoku. O processo de treinamento do modelo e as configurações para obter as imagens necessárias foram desafiadores, mas permitiram explorar a aplicação prática de IA na resolução de problemas do mundo real.

