import torch
import torch.nn as nn

from .condensed_encoder import Encoder
from .condensed_decoder import ImgDecoder




import sys

class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""
    
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode = False):
        """
        Parameters
        ----------
        input_dim: int
            The number of input channels in an image.
        latent_dim: int
            The latent dimension.
        """

        super(VAE, self).__init__()
        
        self.with_logits = with_logits
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode
        self.encoder = Encoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.img_decoder = ImgDecoder(input_dim=1, latent_dim=self.latent_dim, with_logits=self.with_logits)
        
        self.mean_params = Lambda(lambda x: x[:, :self.latent_dim]) # mean parameters
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim:]) # log variance parameters
        
    
    def forward(self, img):
        """ Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        
        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    def forward_test(self, img):
        """ Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        
        # encode
        z = self.encoder(img)

        # reparametrization trick
        mean = self.mean_params(z)
        logvar = self.logvar_params(z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = mean + eps * std

        # decode
        img_recon = self.img_decoder(z_sampled)
        return img_recon, mean, logvar, z_sampled

    
    def encode(self, img):
        """ Do a forward pass of the VAE. Generates a latent vector based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        z = self.encoder(img)
        
        means = self.mean_params(z)
        logvars = self.logvar_params(z)
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(logvars)
        if self.inference_mode:
            eps = torch.zeros_like(eps)
        z_sampled = means + eps * std

        return z_sampled, means, std
    

    def decode(self, z):
        """ Do a forward pass of the VAE. Generates a reconstructed image based on z
        Parameters
        ----------
        z: torch.Tensor
            The latent vector.
        """
        img_recon = self.img_decoder(z)
        if self.with_logits:
            return torch.sigmoid(img_recon)
        return img_recon
    
    def set_inference_mode(self, mode):
        self.inference_mode = mode


if __name__ == "__main__":
    from torchsummary import summary

    LATENT_DIM = 64
    device = torch.device("cpu")
    encoder = Encoder(input_dim=1, latent_dim=LATENT_DIM).to(device)
    summary(encoder, input_size=(1, 270, 480), batch_size=-1, device=device.type)

    decoder = ImgDecoder(input_dim=1, latent_dim=LATENT_DIM).to(device)
    summary(decoder, input_size=(1, LATENT_DIM), batch_size=-1, device=device.type)

    vae = VAE(input_dim=1, latent_dim=LATENT_DIM, with_logits=False)
    summary(vae, input_size=(1, 270, 480), batch_size=-1, device=device.type)