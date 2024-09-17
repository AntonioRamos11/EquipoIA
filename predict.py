import os
import numpy as np

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


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


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def transformLabels(y_test):
    # Si y_test es un vector de etiquetas one-hot codificadas, convierte cada vector en el índice de la clase
    return np.argmax(y_test, axis=1)

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
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Funcion para la precision
def compute_accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == np.argmax(labels, axis=0))
    total_predictions = labels.shape[1]
    accuracy = correct_predictions / total_predictions
    return accuracy


import pickle
#load the model
with open('model.pkl', 'rb') as file:
    W1, b1, W2, b2 = pickle.load(file)

X_test = test_images.T
Y_test = test_labels.T
Y_train = train_labels.T  # Transpone las etiquetas también para que quede como (10, 60000)
X_train = train_images.T
predictions = predict(X_test, W1, b1, W2, b2)
print(predictions[0])
Y_valores = transformLabels(Y_train)
print(predictions[:10])
print(Y_valores[:10])
print('Accuracy: ',compute_accuracy(predictions,Y_test))

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions


def forward_propagation2(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1  # Nota el uso de X.T para alinear dimensiones
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def predict2(image, W1, b1, W2, b2):
    # Asegúrate de que la imagen tenga la forma correcta
    if image.shape != (784,):
        image = image.reshape(1, 784)  # Añadir una dimensión de lote si es necesario
    
    _, _, _, A2 = forward_propagation2(image, W1, b1, W2, b2)
    # Supongamos que A2 es el vector de probabilidades para cada clase
    return np.argmax(A2, axis=1)  # Devuelve la clase con mayor probabilidad

from PIL import Image


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
    img_array = np.array(img)  # Convertir la imagen a un array NumPy
    img_array = img_array.flatten()  # Aplanar la imagen
    img_array = img_array / 255.0  # Normalizar los valores de los píxeles a [0, 1]
    return img_array

# Ejemplo de uso
image_path = 'ejemplos/2.jpeg'

image_path = 'ejemplos/'
elementos_en_carpeta = []
for filename in os.listdir(image_path):
    elementos_en_carpeta.append(filename)


for x in elementos_en_carpeta:
    preprocessed_image = load_and_preprocess_image('ejemplos/'+x)
    predict_image = predict(preprocessed_image, W1, b1, W2, b2)
    print(predict_image[0])