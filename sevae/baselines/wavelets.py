import numpy as np
import pywt

class WaveletTransforms:
    def __init__(self, name='db1', level=7):
        self.image = None
        self.coeffs = None
        self.reconstructed_image = None
        self.name = name
        self.level = level

    def get_wavelet_coefficients(self, image, latent_dims=128):
        self.image = image
        self.coeffs = pywt.wavedec2(image, wavelet=self.name, level=self.level)
        # select top latent_dims coefficients after sorting
        coeff_array, coeff_slices = pywt.coeffs_to_array(self.coeffs)
        Csort = np.sort(np.abs(coeff_array.reshape(-1)))
        threshold = Csort[-latent_dims]
        coeff_array[np.abs(coeff_array) < threshold] = 0
        coeffs_filt_array = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format='wavedec2')
        self.reconstructed_image = pywt.waverec2(coeffs_filt_array, wavelet=self.name)
        return coeff_array, self.reconstructed_image

    def forward_with_latent_dims(self, image, latent_dims=128):
        return self.get_wavelet_coefficients(image, latent_dims)
    
    def get_reconstruction_with_latent_dims(self, latent_space, slices):
        coeffs_filt_array = pywt.array_to_coeffs(latent_space, slices, output_format='wavedec2')
        reconstructed_image = pywt.waverec2(coeffs_filt_array, wavelet=self.name)
        return reconstructed_image