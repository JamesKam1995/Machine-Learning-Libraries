import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
from torchvision.models import vgg19

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    target = target.view(input.size())
    return bce(input.squeeze(), target.squeeze()) # squeeze target to match input

def adversarial_loss(logits_real, logits_fake):
    """
    Computes the adversarial loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the adversarial loss
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    loss_real = bce_loss(logits_real, torch.ones_like(logits_real))
    loss = loss_fake + loss_real
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def VGG_loss(input, target):
    """
    
    Inputs: 
    - input: Pytorch Tensor of shape (N, ) giving scores
    - target: Pytorch Tensor of shape (N, ) containing 0 and 1 giving target

    Returns:
    - A PyTorch Tensor containing the mean VGG loss over the minibatch of input data
    """
    vggloss = vgg19(pretrained=True).features[:36].eval()
    MSE_loss = nn.MSELoss()

    for param in vggloss.parameters():
            param.requires_grad = False
    
    vgg_input = vggloss(input)
    vgg_target = vggloss(target)
    
    return MSE_loss(vgg_input, vgg_target)

def pixel_loss(input, target):
     """
    Inputs: 
    - input: Pytorch Tensor of shape (N, ) giving scores
    - target: Pytorch Tensor of shape (N, ) containing 0 and 1 giving target

    Returns:
    - A PyTorch Tensor containing the mean pixel loss over the minibatch of input data
    """
     return nn.MSELoss(input, target)

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return optimizer

def train_SRGAN(G, D, G_optimizer, D_optimizer, train_loader, vgg_loss, adversarial_loss,
                pixel_loss, show_every=250, num_epochs=10):
    """
    Train SRGAN
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_optimizer, G_optimizer: torch.optim Optimizers to use for training the
      discriminator and generator.
    - train_loader: Yields (lr, hr) image pairs
    - vgg_loss : VGG based perceptual loss function
    - adversarial_loss : BCEWithLogitsLoss for adversarial loss
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """

    for epoch in range(num_epochs):
        for i, (lr, hr) in enumerate(train_loader):
            
            #Train Discriminator
            D_optimizer.zero_grad()

            sr = G(lr)
            real_out = D(hr)
            fake_out = D(sr.detach())
            d_loss = adversarial_loss(real_out, fake_out)
            d_loss.backward()
            D_optimizer.step()

            #Train Generator
            G_optimizer.zero_grad()
            sr = G(lr)
            fake_out = D(sr)

            # Losses: pixel + perceptual + adversarial
            pixel_loss = pixel_loss(sr, hr)
            perceptual_loss = vgg_loss(sr, hr)
            ad_loss = adversarial_loss(fake_out, torch.ones_like(fake_out))
            g_loss = pixel_loss + 0.006 * perceptual_loss + 1e-3 * ad_loss

            g_loss.backward()
            G_optimizer.step()

            # ---------------------
            # Show Progress
            # ---------------------
            if (i + 1) % show_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

