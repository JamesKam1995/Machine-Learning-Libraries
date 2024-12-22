# import libraries
import numpy as np
from neural_network import FCLayer # Linear Layer 
import matplotlib.pyplot as plt # For plotting the loss and accuracy

# The following libraries are loaded for data loading and preprocessing 
from keras.datasets import mnist
from keras.utils import to_categorical

class CreateModel():
    def __init__(self, input_size, output_size, hidden_size):
         self.layer1 = FCLayer(input_size = input_size, output_size = hidden_size[0], activation='relu')
         self.layer2 = FCLayer(input_size = hidden_size[0], output_size = hidden_size[1], activation='relu')
         self.layer3 = FCLayer(input_size = hidden_size[1], output_size = output_size, activation='softmax')

    def forward(self, inputs):
         output1 = self.layer1.forward(inputs)
         output2 = self.layer2.forward(output1)
         output3 = self.layer3.forward(output2)

         return output3
    
    def train(self, inputs, targets, n_epochs, initial_learning_rate, decay, plot_training_results = False):
         #define timesteps
         t = 0

         loss_log = []
         accuracy_log = []

         for epoch in range(n_epochs):
              output = self.forward(inputs=inputs)

              #Calculate the cross entropy loss
              epsilon = 1e-10
              loss =- np.mean(targets * np.log(output + epsilon))

              #calculate the accuract
              predicted_labels = np.argmax(output, axis =1)
              true_labels = np.argmax(targets, axis =1 )
              accuracy = np.mean(predicted_labels == true_labels)

              #backward
              output_grad = 6 * (output - targets) / output.shape[0]
              t += 1 
              learning_rate = initial_learning_rate / (1 + decay * epoch)
              grad_3 = self.layer3.backward(output_grad, learning_rate, t)
              grad_2 = self.layer2.backward(grad_3, learning_rate, t)
              grad_1 = self.layer1.backward(grad_2, learning_rate, t)

              print(f"Epoch {epoch} // Loss: {loss} // Accuracy: {accuracy}")

              #Add the loss and accuracy to the list
              if plot_training_results == True:
                   loss_log.append(loss)
                   accuracy_log.append(accuracy)

         if plot_training_results == True:
              plt.plot(range(n_epochs), loss_log, label="Training Loss")
              plt.xlabel("Epoch")
              plt.ylabel("Loss")
              plt.title("Training Loss Curve")
              plt.legend()
              plt.show()

              plt.plot(range(n_epochs), accuracy_log, label="Training Loss")
              plt.xlabel("Epoch")
              plt.ylabel("Accuracy")
              plt.title("Training Loss Curve")
              plt.legend()
              plt.show()  

if __name__ == "__main__":   
    # Define hyperparameters for training 
    INPUT_SIZE = 784
    HIDDEN_SIZE = [512, 512]
    OUTPUT_SIZE = 10            
                
                
    # load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    x_train = x_train.reshape((60000, 784))

    # Normalize 
    x_train = x_train.astype("float32") / 255.0

    # Preprocess the labels 
    y_train = to_categorical(y_train)

    # Create the Neural Network model
    nn = CreateModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
    nn.train(x_train, y_train, initial_learning_rate=0.001, decay=0.001, n_epochs=100, plot_training_results=True)       
            

