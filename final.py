import numpy as np
import os
from PIL import Image

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Load Functions
def load_images(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = images / 255.0
        images = images.reshape(num_images, rows * cols)
        return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

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

        

# Initialize Weights
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.05
    W2 = np.random.randn(output_size, hidden_size) * 0.05
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Forward Propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward Propagation
def backpropagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update Parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Compute Loss
def compute_loss(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y)
    loss = -np.sum(logprobs) / m
    return loss

# Compute Accuracy
def compute_accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == np.argmax(labels, axis=0))
    total_predictions = labels.shape[1]
    accuracy = correct_predictions / total_predictions
    return accuracy

# Training
def train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(A2, Y)
        dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        predictions = predict(X, W1, b1, W2, b2)
        accuracy = compute_accuracy(predictions, Y)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return W1, b1, W2, b2

# Prediction
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Transform Labels from one-hot encoding
def transform_labels(one_hot_labels):
    return np.argmax(one_hot_labels, axis=0)

# Load and Preprocess Image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.flatten()
    img_array = img_array / 255.0
    return img_array

# Load dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist(
    'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
)
"""
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Example train image: {train_images[0]}")
print(f"Example train label: {train_labels[0]}")


"""




# Training parameters
input_size = 784  # 28x28 pixels
hidden_size = 32  # Size of hidden layer
output_size = 10  # Digits (0-9)
epochs = 50
learning_rate = 0.001

X_train = train_images.T
Y_train = train_labels.T

x_train,x_test,y_train,y_test = X_train[:60000],X_train[60000:],Y_train[:60000],Y_train[60000:]

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10,4))
sns.histplot(data=np.int8(Y_train),binwidth=0.45,bins=11)
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9],labels=[0,1,2,3,4,5,6,7,8,9])
plt.xlabel('Class')
plt.title('Distribution of Samples')
plt.show()



# Train the model
W1, b1, W2, b2 = train(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate)

# Predict on test data
X_test = test_images.T
Y_test = test_labels.T
predictions = predict(X_test, W1, b1, W2, b2)

# Compute accuracy on test data
accuracy = compute_accuracy(predictions, Y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Example of prediction on a new image
def predict_image(image_path, W1, b1, W2, b2):
    img_array = load_and_preprocess_image(image_path)
    img_array = img_array.reshape(1, 784).T  # Reshape for prediction
    prediction = predict(img_array, W1, b1, W2, b2)
    return prediction

# Predicting a new image
image_path = 'ejemplos/0.jpeg'  # Replace with your image path
prediction = predict_image(image_path, W1, b1, W2, b2)
print(f'Predicted Class for the image: {prediction[0]}')

# Save model
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump((W1, b1, W2, b2), file)
