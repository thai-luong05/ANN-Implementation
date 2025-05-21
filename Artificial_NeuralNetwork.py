import math, random, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tensorflow as tf
# NOTE: THIS IS FOR DIGIT RECOGNITION
class ANN():
    def __init__(self):
        n_in = 784 #28x 28
        n_out = 10
        #xavier, glorot initialization
        self.weight1 = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), size=(n_in, n_out)) #size should be 10 vector
        self.weight2 = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), size=(10, 10)) #for the second layer hidden
        self.bias1 = np.array([0] * n_out, dtype=float).reshape(1, 10)
        self.bias2 = np.array([0] * n_out, dtype=float).reshape(1, 10)
        
    def sigmoid(self, x):
        return 1/(1+math.e**(-x))
    
    def error(self, output, desire):
        total = 0
        for i in range(10):
            total += (output[i] - desire[i])**2
        return total / 2

    def train(self, entries, data, epochs): #train perhaps?
        rate = 0.01
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in entries:
                input = data[i][0]
                desired_output =data[i][1]

                z1 = input @ self.weight1 + self.bias1
                out1 = self.sigmoid(z1)
                z2 = out1 @ self.weight2 + self.bias2
                out2 = self.sigmoid(z2)  
                loss = np.mean((out2 - desired_output)**2)
                epoch_loss += loss #loss for each data

                #calculating weights using chain rule
                de_do2 = out2 - desired_output
                do2_dz2 = out2 * (1 - out2)
                #delta: error per neuron
                delta2 = de_do2 * do2_dz2
                dz2_dw2 = out1.T

                de_dw2 = dz2_dw2 @ delta2
                #sum all the gradients in order for the last data not to be dominant, also sum across batch to adapt to the overall data
                de_db2 = np.sum(delta2, axis = 0, keepdims = True)

                delta1 = (delta2 @ self.weight2.T) * out1 * (1 - out1) #de_dz1
                de_dw1 = input.T @ delta1
                de_db1 = np.sum(delta1, axis = 0, keepdims = True)

                #update weights
                self.weight1 -= rate * de_dw1
                self.weight2 -= rate * de_dw2
                self.bias1 -= rate * de_db1
                self.bias2 -= rate * de_db2
            epoch_losses.append(epoch_loss / len(data))
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(data):.5f}") #avg loss across all data
        return epoch_losses
    
    def retest(self, input):
        z1 = input @ self.weight1 + self.bias1
        out1 = self.sigmoid(z1)
        z2 = out1 @ self.weight2 + self.bias2
        out2 = self.sigmoid(z2)
        
        predicted_class = np.argmax(out2)
        return predicted_class      

#train multiple times to see accuracy
accuracy = []
for trials in range(10):
    print(f"trial {trials + 1}")
    network = ANN()
    data = {}
    size = -1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #to gray scale
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    for i in range(y_train.shape[0]):
        rgb = x_train[i].reshape(1, 784)
        desired_output = [0] * 10
        desired_output[y_train[i]] = 1
        desired_output = np.array(desired_output).reshape(1, 10)
        size += 1
        data[size] = (rgb, desired_output)

    entries = [i for i in data.keys()]
    #hsuffle before train
    random.shuffle(entries)
    #train
    avg_loss = network.train(entries, data, epochs=20)
    plt.figure()
    plt.plot(avg_loss)
    plt.title(f'losses during epochs for trial {trials + 1}')
    plt.ylabel("Average loss")
    plt.xlabel("Epochs #")
    #test
    total = 0
    correct = 0
    for i in range(y_test.shape[0]):
        number = y_test[i]
        rgb = x_test[i].reshape(1, 784)
        result = network.retest(rgb)
        if result == number:
            correct += 1
        total += 1
    accuracy.append(correct / total)

plt.figure()
plt.title("Accuracy for each trials of training")
plt.plot(accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Trials #")
plt.show()