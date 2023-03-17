import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Valores de los pesos sinápticos y los sesgos
w11 = 0.6
w12 = 0.3
b1 = 0.5
w21 = 0.4
w22 = 0.8
b2 = -0.5
w31 = 0.5
w32 = -0.4
b3 = 0.4

# Datos de entrada
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

# Salida deseada
y_deseado = np.array([0, 1, 1, 0])

# Calcular la salida de la red neuronal para cada par de entradas
for i in range(len(x1)):
    # Capa oculta
    z1 = sigmoid(w11 * x1[i] + w21 * x2[i] + b1)
    z2 = sigmoid(w12 * x1[i] + w22 * x2[i] + b2)
    # Capa de salida
    y = sigmoid(w31 * z1 + w32 * z2 + b3)
    # Mostrar los resultados
    print(f"Entradas: ({x1[i]}, {x2[i]}) - Salida: {y:.2f} - Salida deseada: {y_deseado[i]}")
