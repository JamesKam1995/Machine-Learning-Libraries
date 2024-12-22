import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(CustomNet, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # First layer with ReLU activation
        out1 = F.relu(self.fc1(x))
        
        # Second layer with ReLU activation
        out2 = F.relu(self.fc2(out1))
        
        # Skip connection: add the output of the first layer to the second
        out2 += out1
        
        # Third layer (output layer)
        out3 = self.fc3(out2)
        
        return out3

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define dimensions
input_dim = 3 * 32 * 32  # Example input dimension for CIFAR-10
hidden_dim1 = 1024
hidden_dim2 = 512
output_dim = 10  # CIFAR-10 has 10 classes

# Create the model
model = CustomNet(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

# Define an example input
example_input = torch.randn(5, 3, 32, 32).to(device)  # Batch of 5, 3x32x32 images

# Forward pass
output = model(example_input)
print(output)