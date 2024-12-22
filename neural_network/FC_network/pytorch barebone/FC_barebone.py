import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class Three_layer_FC_barebone:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # Initialize weights and biases
        self.w1 = self._random_weight((input_dim, hidden_dim1))
        self.b1 = torch.zeros(hidden_dim1, device=self.device, dtype=self.dtype)

        self.w2 = self._random_weight((hidden_dim1, hidden_dim2))
        self.b2 = torch.zeros(hidden_dim2, device=self.device, dtype=self.dtype)

        self.w3 = self._random_weight((hidden_dim1, output_dim))
        self.b3 = torch.zeros(output_dim, device=self.device, dtype=self.dtype)

    def forward_pass(self, x):
        x = self._flatten(x)
        x = F.relu(x.mm(self.w1) + self.b1)
        x = F.relu(x.mm(self.w2) + self.b2)
        x = x.mm(self.w3) + self.b3
        return x
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
    
    def train(self, loader_train, loader_val, learning_rate = 1e-2, num_epochs=100, print_every=100):
        for epoch in range(num_epochs):
            for t, (x, y) in enumerate(loader_train):
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)

                scores = self.forward_pass(x)
                loss = F.cross_entropy(scores, y)
                loss.backward()

                with torch.no_grad():
                    for param in self.parameters():
                        param -= learning_rate * param.grad
                        param.grad.zero()

                if t % print_every == 0:
                    print('Epoch %d, iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    self.check_accuracy(loader_val)
                    print()
    
    def check_accuracy(self, loader):
        num_correct = num_samples = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.int64)
                scores = self.forward_pass(x, params)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
                
    def _random_weight(self, shape):
        """
        Create random Tensors for weights; setting requires_grad=True means that we
        want to compute gradients for these Tensors during the backward pass.
        We use Kaiming normalization: sqrt(2 / fan_in)
        """
        if len(shape) == 2:
            fan_in = shape[0]
        else:
            fan_in = np.prod(shape[1:])

        w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
        w.requires_grad = True
        return w
    
    def _flatten(self, x):
        N = x.shape[0]
        return x.view(N, -1)