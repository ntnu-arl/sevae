
import os
import glob
import pickle
import sys
import torch


from sevae.networks.VAE.vae import VAE




LOAD_MODEL= True


# Data Path
BASE_PATH = "/home/arl/MihirK/DATA"

# VAE Hyperparams
LATENT_DIM = 128
LOAD_MODEL = True

MAX_DEPTH = 10.0
MIN_DEPTH = 0.2

global_pickle_counter = 0

def read_pickle_files(depth_pickle_file, append_filename):
    print("Reading pickle files: ", depth_pickle_file)
    if not os.path.isfile(depth_pickle_file):
        print("Depth file does not exist")
        return

    with open(depth_pickle_file, 'rb') as f:
        depth_img_list = pickle.load(f)

    num_env = 1
    num_timesteps = 0

    depth_latents = []
    depth_flipped_latents = []

    for episode in range(len(depth_img_list)):
        depth_latents.append([])
        depth_flipped_latents.append([])
        for timestep in range(len(depth_img_list[episode])):
            depth_image = depth_img_list[episode][timestep].copy()
            depth_image_torch = torch.from_numpy(depth_image).squeeze(-1).to("cuda")
            depth_image_torch = depth_image_torch.unsqueeze(0).unsqueeze(0)
            depth_image_flipped = torch.flip(depth_image_torch, [3])

            depth_image_torch = torch.clamp(depth_image_torch, 0, MAX_DEPTH)
            depth_image_flipped = torch.clamp(depth_image_flipped, 0, MAX_DEPTH)
            depth_image_torch = depth_image_torch / MAX_DEPTH
            depth_image_flipped = depth_image_flipped / MAX_DEPTH

            depth_image[depth_image < MIN_DEPTH/MAX_DEPTH] = -1.0
            depth_image_flipped[depth_image_flipped < MIN_DEPTH/MAX_DEPTH] = -1.0

            # get the latent vector
            latent_vector_straight_image = model.encode(depth_image_torch)[1].detach().cpu().numpy().squeeze(0)
            latent_vector_flipped_image = model.encode(depth_image_flipped)[1].detach().cpu().numpy().squeeze(0)

            depth_latents[episode].append(latent_vector_straight_image)
            depth_flipped_latents[episode].append(latent_vector_flipped_image)
            num_timesteps += 1

    # print(len(depth_img_list), num_timesteps)
    # save pickle
    depth_latent_filename = depth_pickle_file.replace("di_dump", "di_latent_"+append_filename)
    depth_flipped_latent_filename = depth_pickle_file.replace("di_dump", "di_flipped_latent_"+append_filename)
    print("Saving pickles to: ", depth_latent_filename, depth_flipped_latent_filename)
    with open(depth_latent_filename, 'wb') as f:
        pickle.dump(depth_latents, f)
    with open(depth_flipped_latent_filename, 'wb') as f:
        pickle.dump(depth_flipped_latents, f)
    print("Saved pickle to VAE compatible format")



if __name__ == "__main__":
    # parse over folder and get all di_dump*.p pickles and rgb_dump*.p pickles
    # read experiment name to load weights from 
    if len(sys.argv) < 3:
        print("Using IROS 2023 submission model weights")
        experiment_name  = "NTNU_thin_ft_labeled_invald_px_test"
    else:
        experiment_name = sys.argv[2]

    LOAD_MODEL_FILE = os.path.join(BASE_PATH, "experiments/"+experiment_name+"/models/"+experiment_name+"_LD_128_epoch_39.pth")
    

    # read folder name from first argument 
    folder_name = sys.argv[1]

    print("Loading di_dump pickles from folder: ", folder_name)

    # print("Checking folder: ", os.join(folder_name))

    if len(sys.argv) < 4:
        print("Appending filename with \"test\"")
        append_filename = "test"
    else:
        append_filename = sys.argv[3]

    print("Appending filename of latent pickles with: ", append_filename)

    print("Number of pickle pairs to process: ", len(glob.glob(folder_name + "/di_dump*.p")))

    model = VAE(latent_dim=LATENT_DIM, with_logits=True)

    print("Loading model from file: ", LOAD_MODEL_FILE)
    if LOAD_MODEL:
        print("Loaded state dict")
        loaded_dict = torch.load(LOAD_MODEL_FILE)
        for key in list(loaded_dict.keys()):
            if "module." in key:
                loaded_dict[key.replace("module.", "")] = loaded_dict[key]
                del loaded_dict[key]
        print(loaded_dict.keys())
        model.load_state_dict(loaded_dict)
    
    model.to("cuda")
    print("[DONE] Loading model from file: ", LOAD_MODEL_FILE)

    files = glob.glob(folder_name + "/di_dump*.p")
    pickles_list = []
    # get all files in the folder having rgb_dump in the name
    for file in glob.glob(folder_name + "/di_dump*.p"):
        depth_filename = file
        print("Reading files: ", depth_filename)
        # check if files exist 
        if not os.path.isfile(depth_filename):
            print("Depth file does not exist")
            continue
        pickles_list.append(depth_filename)
    
    pickles_list.sort()
    print("Number of pickles to process: ", len(pickles_list))
    for pickle_name in pickles_list:
        read_pickle_files(pickle_name, append_filename)