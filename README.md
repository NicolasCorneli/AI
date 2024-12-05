# 🧩 IA para Resolução de Sudoku

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
   - Durante o treinamento, a precisão foi monitorada continuamente, e ajustes foram feitos conforme necessário para aprimorar a performance do modelo.

5. **Reajuste e Fine-Tuning**:
   - O modelo passou por um processo de fine-tuning, onde hiperparâmetros foram ajustados e a arquitetura foi refinada para otimizar a aprendizagem dos números. Isso permitiu que o modelo identificasse os números de forma mais eficiente e precisa, melhorando seu       
     desempenho na resolução do Sudoku.

## 📋 Requisitos Técnicos

- **Bibliotecas**:
  - TensorFlow
  - Responsabilidade: O TensorFlow é uma biblioteca de aprendizado de máquina usada para criar, treinar e testar modelos de IA. No código, ele é utilizado para construir e treinar a rede neural convolucional (CNN) que faz o reconhecimento dos números nas células do         Sudoku. O TensorFlow gerencia a criação do modelo, o fluxo de dados, o cálculo de gradientes e a otimização dos pesos durante o treinamento.

  - Keras
  - OpenCV
  - NumPy
  - Imutils

- **Ferramentas**:
  - Python 3.x
  - TensorFlow e Keras para treinamento e construção do modelo de IA.
  - OpenCV para processamento de imagens.

## ⚙️ Como Usar

### Passos para Rodar o Projeto

1. **Instalação das Dependências**:
   - Clone o repositório:
     ```bash
     git clone <URL_DO_REPOSITORIO>
     cd <diretório_do_repositorio>
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

Ao executar o script, a IA deve ser capaz de detectar e resolver a grade de Sudoku presente em uma imagem, mostrando o resultado final diretamente sobre a imagem original.

## 📝 Conclusões

Este projeto demonstra como é possível utilizar redes neurais convolucionais para resolver problemas lógicos simples, como o Sudoku. O processo de treinamento e ajuste do modelo foi desafiador, mas permitiu explorar a aplicação prática de IA na resolução de problemas do mundo real.

## 📝 Licença

Este projeto é licenciado sob a [Licença MIT](LICENSE).
