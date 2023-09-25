from scipy.fft import fft2, ifft2
import numpy as np

class FFT:
    def __init__(self):
        self.image = None
        self.fft = None
        self.reconstructed_image = None
    
    def get_image_reconstruction_with_compressed_dimensions(self, image, latent_dims=128):
        pixels = image.shape[0]*image.shape[1]
        self.fft = fft2(image)

        self.thresh_fft = self.fft.copy()
        self.fft_magnitude = np.abs(self.fft).flatten()
        self.fft_magnitude.sort()
        threshold = self.fft_magnitude[-latent_dims]
        self.thresh_fft[np.abs(self.fft) < threshold] = 0
        self.reconstructed_image = ifft2(self.thresh_fft).real # Check if real is to be used or abs
        return self.fft, self.reconstructed_image
    
    def forward(self, image, latent_dims=128):
        return self.get_image_reconstruction_with_compressed_dimensions(image, latent_dims)[1]