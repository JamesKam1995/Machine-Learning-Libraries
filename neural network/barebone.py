import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


import torch.nn.functional as F  # useful stateless functions

class Two_Layer_Barebone:
  def two_layer_fc(x, params):
      """
      A fully-connected neural networks; the architecture is:
      NN is fully connected -> ReLU -> fully connected layer.
      Note that this function only defines the forward pass;
      PyTorch will take care of the backward pass for us.

      The input to the network will be a minibatch of data, of shape
      (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
      and the output layer will produce scores for C classes.

      Inputs:
      - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
        input data.
      - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
        w1 has shape (D, H) and w2 has shape (H, C).

      Returns:
      - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
        the input data x.
      """
      # first we flatten the image
      x = flatten(x)  # shape: [batch_size, C x H x W]

      w1, w2 = params

      # Forward pass: compute predicted y using operations on Tensors. Since w1 and
      # w2 have requires_grad=True, operations involving these Tensors will cause
      # PyTorch to build a computational graph, allowing automatic computation of
      # gradients. Since we are no longer implementing the backward pass by hand we
      # don't need to keep references to intermediate values.
      # you can also use `.clamp(min=0)`, equivalent to F.relu()

      x = F.re(x.mm(w1))
      x = x.mm(w2)
      return x

  def random_weight(shape):
      if len(shape) == 2:
          fan_in = shape[0]

      else:
          fain_in = np.prod(shape[1:])
      w = torch.randn(shape, device=devie, dtype=dtype) * np.sqrt(2. / fan_in)
      w.requires_grad = True
    
  def zero_weight(shape):
      return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
  
  def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

    def train_part2(model_fn, params, learning_rate):
      """
      Train a model on CIFAR-10.

      Inputs:
      - model_fn: A Python function that performs the forward pass of the model.
        It should have the signature scores = model_fn(x, params) where x is a
        PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
        model weights, and scores is a PyTorch Tensor of shape (N, C) giving
        scores for the elements in x.
      - params: List of PyTorch Tensors giving weights for the model
      - learning_rate: Python scalar giving the learning rate to use for SGD

      Returns: Nothing
      """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight(channel_1)
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight(channel_2)
fc_w = random_weight((channel_2 * 32 * 32, 10))
fc_b = zero_weight(10)

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)

