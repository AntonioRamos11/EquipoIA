import numpy as np

# Función para cargar las imágenes desde un archivo .idx
def load_images(filename):
    with open(filename, 'rb') as f:
        # Leer el encabezado
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')

        # Leer el contenido de las imágenes
        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = images / 255.0  # Normalizamos los valores a [0, 1]
        images = images.reshape(num_images, rows * cols)  # Redimensionamos las imágenes a (num_images, 28*28)
        return images

# Función para cargar las etiquetas desde un archivo .idx
def load_labels(filename):
    with open(filename, 'rb') as f:
        # Leer el encabezado
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')

        # Leer las etiquetas
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# Función para convertir etiquetas a one-hot encoding
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

import os
# Función general para cargar el conjunto de datos MNIST
def load_mnist(train_images_file, train_labels_file, test_images_file, test_labels_file):
    # Cargar datos de entrenamiento

    train_images = load_images(os.getcwd() + "/mnis_dataset/" + train_images_file +"/"+train_images_file)
    print((os.getcwd() + "/mnis_dataset/" + train_images_file))
    
    train_labels = load_labels(os.getcwd() + "/mnis_dataset/" + train_labels_file+ "/"+train_labels_file)
    
    # Cargar datos de prueba
    test_images = load_images(os.getcwd() + "/mnis_dataset/" + test_images_file + "/"+test_images_file)
    test_labels = load_labels( os.getcwd() + "/mnis_dataset/" + test_labels_file + "/"+test_labels_file)
    
    # One-hot encoding para etiquetas
    train_labels_one_hot = one_hot_encode(train_labels, 10)
    test_labels_one_hot = one_hot_encode(test_labels, 10)
    
    return (train_images, train_labels_one_hot), (test_images, test_labels_one_hot)

# Uso de la función
(train_images, train_labels), (test_images, test_labels) = load_mnist(
    'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
)


import numpy as np

# Función de activación sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoidal
def sigmoid_derivative(x):
    return x * (1 - x)

# Función de activación softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

# Inicializar pesos aleatorios
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Propagación hacia adelante
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Retropropagación
def backpropagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Actualización de los pesos
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

# Función de pérdida
def compute_loss(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y)
    loss = -np.sum(logprobs) / m
    return loss

# Entrenamiento del modelo
def train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(A2, Y)
        dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss}")

    return W1, b1, W2, b2

# Predicción
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Cargar tus datos (MNIST o de una imagen dibujada)
# Por simplicidad, asumimos que X es la entrada y Y son las etiquetas (ambas ya preprocesadas)

# Tamaños de la red
input_size = 784  # 28x28 píxeles
hidden_size = 128  # Tamaño de la capa oculta
output_size = 10   # Dígitos (0-9)

# Entrenar la red
epochs = 1000
learning_rate = 0.01



X_train = train_images.T  # Transpone para que quede como (784, 60000)
Y_train = train_labels.T  # Transpone las etiquetas también para que quede como (10, 60000)

# Ahora puedes entrenar el modelo correctamente
W1, b1, W2, b2 = train(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate)
# Para una nueva imagen, realiza la predicción
# predicción = predict(X_test, W1, b1, W2, b2)

X_test = test_images.T
predictions = predict(X_test, W1, b1, W2, b2)

