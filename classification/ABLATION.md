# Estudio de ablación — Clasificación de sonidos respiratorios

## 1. Selección de representaciones 2D
1. STFT (espectrograma)
2. CWT (escalograma)
3. WSST (synchrosqueezed scalogram)
4. STFT + CWT
5. STFT + WSST
6. CWT + WSST
7. STFT + CWT + WSST

## 2. Selección de representación 1D
1. Señal original — CNN
2. IMF — CNN (n canales)
3. Reconstrucción IMF — CNN (1 canal)
4. IMF — 1D CNN (1 canal, late fusion)
5. RNN (p. ej. GRU)

## 3. Selección de representación de entrada
1. Solo 2D (mejor opción de la etapa 1)
2. Solo 1D (mejor opción de la etapa 2)
3. 2D + 1D

## 4. Modificación de la CNN
1. Sin modificaciones (mejor opción de la etapa 3)
2. Residual
3. Dense
4. SE/CBAM (Attention Convolutional Neural Network)
## 5. Late fusion
1. Sin atención (mejor opción de la etapa 4)
2. Con atención

## 6. Clasificador
1. Multilayer perceptron (mejor opción de la etapa 5)
2. LR
3. SVM
4. LGB

## 7. Tipo de clasificación
1. Multiclase (mejor opción de la etapa 6)
2. Jerárquica (Normal vs. adventicio; tipo de adventicio)