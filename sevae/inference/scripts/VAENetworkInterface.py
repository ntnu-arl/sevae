import time
import torch

import sevae
from sevae.networks.VAE.vae import VAE

# collision_repreesntation path
COLL_REP_PATH = sevae.__path__[0]
# weights path
# weights/seVAE_weights/dragvoll_dataset_test_6_LD_128_epoch_39.pth
# WEIGHTS_PATH = COLL_REP_PATH + "/weights/seVAE_weights/seVAE_LD_128.pth"

WEIGHTS_PATH = COLL_REP_PATH + "/weights/seVAE_weights/seVAE_evo_LD_128.pth"


class VAENetworkInterface():
    def __init__(self, latent_space_dims=128, device="cuda:0"):
        self.latent_space = latent_space_dims
        self.device = device

        # Load networks in constructor
        try:
            self.vae = VAE(input_dim=1, latent_dim=self.latent_space)
            dict = torch.load(WEIGHTS_PATH)
            self.vae.load_state_dict(dict, strict=True)
            self.vae = self.vae.to(self.device)
        except:
            print("Could not load networks")
            raise Exception("Could not load networks")
        print("Loading: " + WEIGHTS_PATH)
        print("VAE MODEL LOADED TO DEVICE: ", self.device)

    def forward(self, image_numpy, calculate_reconstruction=False):
        self.start_time = time.time()
        with torch.no_grad():
            torch_image = torch.from_numpy(image_numpy).float().to(self.device)
            torch_image = torch.clamp(torch_image, 0.0, 1.0)
            z_sampled, means, *_ = self.vae.encode(torch_image.view(1, 1, 270, 480))
            reconstructed_image = 0
            if calculate_reconstruction:
                reconstructed_image = self.vae.decode(means).cpu().numpy()
        self.end_time = time.time()
        return means.cpu().numpy(), reconstructed_image, 1000*(self.end_time - self.start_time)
    
    def get_compute_time(self):
        return (self.end_time - self.start_time)*1000
