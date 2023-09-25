import os, sys
from random import shuffle

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from networks.VAE.vae import VAE
from networks.Loss.running_loss import RunningLoss
from networks.Loss.loss_functions import *
# from datasets.depth_image_dataset import DepthImageDataset, collate_batch
from datasets.depth_semantic_image_dataset import IMAGES_PER_TF_RECORD, DepthSemanticImageDataset, collate_batch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision


# import module for random sampling
import random


import tensorflow as tf
import yaml
import inspect




#  Use argument parser to set arguments of experiment name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="default")
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--load_model_path", type=str, default=None)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("GPU error")
        print(e)


device = torch.device("cuda")
device0 = torch.device("cuda:0")


# VAE Hyperparams
LATENT_DIM = 128
NUM_EPOCHS = 40
ONE_TFRECORD = False
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

SAVE_MODEL = False
LOAD_MODEL = False
SAVE_INTERVAL = 10  # Save the model every 10 batches

# Model Paths
BASE_PATH = "/home/arl/MihirK/DATA"
EXPERIMENT_NAME = "default"
EXPERIMENT_BASE_PATH = os.path.join(BASE_PATH, EXPERIMENT_NAME)
SAVE_MODEL_FILE = os.path.join(BASE_PATH, "models") # "_epochxx.pth" appended in training
LOAD_MODEL_FILE = os.path.join(BASE_PATH, "vae_models/final_BCE__weighted_100_canny_mae2__weighted_10_depth_100.pth")

# Data Path
TFRECORD_FOLDER = os.path.join(BASE_PATH, "datasets")
TFRECORD_TEST_FOLDER = os.path.join(TFRECORD_FOLDER, "test")

MULTI_GPU = True
FILL_UNDEFINED_PIXELS_WITH_NEGATIVE_VALUES = True

ADD_NOISE_TO_INPUT = False



MAX_DEPTH = 10.0
MIN_DEPTH = 0.15


def make_grid_for_tensorboard(images_list, n_grids=2):
    joined_images = []
    [joined_images.extend(images[:n_grids]) for images in images_list]
    return torchvision.utils.make_grid(joined_images, nrow=n_grids, padding=5)


def get_noise(means, std_dev, const_multiplier):
    '''
    
    '''
    return const_multiplier*torch.normal(means, std_dev)

def process_for_training(input_image, filled_input_image, semantic_input_image):
    '''
    Function to process the input image for training
    '''
    processed_input_image = input_image.clone()
    processed_filled_input_image = filled_input_image.clone()
    processed_semantic_input_image = semantic_input_image.clone()

    processed_input_image[processed_input_image > MAX_DEPTH] = MAX_DEPTH
    processed_input_image[processed_input_image < MIN_DEPTH] = -1.0
    processed_input_image = processed_input_image / MAX_DEPTH
    processed_input_image[processed_input_image < 0] = -1.0

    processed_filled_input_image = torch.clamp(processed_filled_input_image, min=0, max=MAX_DEPTH)
    processed_filled_input_image[processed_filled_input_image < MIN_DEPTH] = MAX_DEPTH
    processed_filled_input_image = processed_filled_input_image / MAX_DEPTH

    processed_semantic_input_image = torch.where(input_image < 0.99*MAX_DEPTH, semantic_input_image, torch.zeros_like(semantic_input_image))

    # Do not have any semantic less thatsemaantic number 9
    processed_semantic_input_image[processed_semantic_input_image < 9] = 0

    if FILL_UNDEFINED_PIXELS_WITH_NEGATIVE_VALUES:
        image_to_reconstruct = torch.where(input_image > MIN_DEPTH, processed_input_image, -1.0*torch.ones_like(input_image))
    
    processed_input_image_with_noise = processed_input_image.clone()
    if ADD_NOISE_TO_INPUT:
        std_dev = torch.zeros_like(input_image)
        std_dev[:] = input_image*MIN_DEPTH/MAX_DEPTH # interpret this as: std_dev at max depth = 0.15m. std_dev at min depth = 0.0m. linearly increasing in between.
        processed_input_image_with_noise = image_to_reconstruct + get_noise(torch.zeros_like(image_to_reconstruct), torch.ones_like(image_to_reconstruct), std_dev)
        processed_input_image_with_noise[processed_input_image_with_noise > 1.0] = 1.0
        processed_input_image_with_noise[input_image < 0] = -1.0

    return processed_input_image_with_noise, image_to_reconstruct, processed_input_image, processed_filled_input_image, processed_semantic_input_image


def train_model(model, train_dataset_loader, test_dataset_loader, epochs, writer, batch_size, loss_fn, optimizer, save_interval=10):
    '''
    Function to train the given input model based on the given data and update the tensorboard
    '''
    torch.autograd.set_detect_anomaly(True)
    # Initialize the model
    model.train()
    # Initialize the loss and the optimizer
    loss_meter = RunningLoss(batch_size)
    # Initialize the number of batches
    num_batches = int(len(train_dataset_loader)/ batch_size)

    num_test_batches = int(len(test_dataset_loader)/ batch_size)
    # Initialize the number of epochs
    for epoch in range(epochs):
        if MULTI_GPU:
            model.module.set_inference_mode(False)
        else:
            model.set_inference_mode(False)
        model.train()
        # Initialize the time counter
        epoch_start_time = time.time()
        # Initialize the number of batches
        for batch_idx, (num_envs, images_per_env, depth_data, filtered_data, semantic_data) in enumerate(train_dataset_loader):
            batch_start_time = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            depth_data = depth_data.to(device).unsqueeze(1)
            filtered_data = filtered_data.to(device).unsqueeze(1)
            semantic_data = semantic_data.to(device).unsqueeze(1)

            noisy_image, depth_data_to_reconstruct, filtered_data, filled_filtered_data, semantic_data = process_for_training(depth_data, filtered_data, semantic_data)

            # Forward pass
            reconstructed_image, means, log_vars, sampled_latent_vars  = model(noisy_image)
            clamped_g_cam_map = None
            loss, kld_loss = loss_fn(depth_data_to_reconstruct, reconstructed_image, semantic_data, clamped_g_cam_map, means, log_vars)
            
            # Update the loss meter
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the tensorboard
            if batch_idx % save_interval == 0 and batch_idx is not 0:
                writer.add_scalar('Train/Loss', loss.item()/BATCH_SIZE, epoch * num_batches + batch_idx)
                writer.add_scalar('Train/KLD Loss', kld_loss.item()/BATCH_SIZE, epoch * num_batches + batch_idx)
                print(f"[TRAINING] Epoch: {epoch}/{epochs} Batch: {batch_idx}/{num_batches} Avg. Train Loss: {loss.item()/BATCH_SIZE:.4f}, KL Div Loss.: {kld_loss.item()/BATCH_SIZE:.4f}"\
                    f"Time: {time.time() - batch_start_time:.2f}s, Est. time remaining: {(num_batches - batch_idx)*avg_iter_time :.2f}s")

                # add image to the tensorboard
                grid = make_grid_for_tensorboard([filled_filtered_data, depth_data_to_reconstruct, noisy_image, torch.sigmoid(reconstructed_image), semantic_data], n_grids=4)
                writer.add_image('training/images', grid, global_step=epoch*num_batches + batch_idx)
                if batch_idx % (5*save_interval) == 0:
                    torchvision.utils.save_image(grid, EXPERIMENT_BASE_PATH + "/training_images" + "/" + EXPERIMENT_NAME + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png")

        # Print the statistics
        print('Epoch: %d, Loss: %.4f, Time: %.4f' %
              (epoch, loss_meter.avg, time.time() - epoch_start_time))
        # Reset the loss meter
        loss_meter.reset()
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(
            EXPERIMENT_BASE_PATH+"/models", '%s_LD_%d_epoch_%d.pth' % (EXPERIMENT_NAME, LATENT_DIM,  epoch)))
        print("[DONE] Savng model at ", EXPERIMENT_BASE_PATH+"/models")

        # # Evaluate the model
        model.eval()
        if MULTI_GPU:
            model.module.set_inference_mode(True)
        else:
            model.set_inference_mode(True)
        for batch_idx, (num_envs, images_per_env, depth_data, filtered_data, semantic_data) in enumerate(test_dataset_loader):
            model.zero_grad()
            optimizer.zero_grad()
            # Process for testing
            
            depth_data = depth_data.to(device).unsqueeze(1)
            filtered_data = filtered_data.to(device).unsqueeze(1)
            semantic_data = semantic_data.to(device).unsqueeze(1)

            noisy_image, depth_data_to_reconstruct, filtered_data, filled_filtered_data, semantic_data = process_for_training(depth_data, filtered_data, semantic_data)

            # Forward pass
            reconstructed_image, means, log_vars, sampled_latent_vars  = model(noisy_image)
            clamped_g_cam_map = None
            loss, kld_loss = loss_fn(depth_data_to_reconstruct, reconstructed_image, semantic_data, clamped_g_cam_map, means, log_vars)
            
            # Update the loss meter
            loss_meter.update(loss.item())
            avg_iter_time = (time.time() - epoch_start_time) / (batch_idx + 1)

            grid = make_grid_for_tensorboard([filled_filtered_data, depth_data_to_reconstruct, noisy_image, torch.sigmoid(reconstructed_image), semantic_data], n_grids=4)
            writer.add_image('testing/images', grid, global_step=epoch)
            if batch_idx % (save_interval) == 0 and batch_idx is not 0:
                print(f"[TESTING] Epoch: {epoch}/{epochs} Batch: {batch_idx}/{num_test_batches} Avg. Train Loss: {loss.item()/BATCH_SIZE:.4f}, KL Div Loss.: {kld_loss.item()/BATCH_SIZE:.4f}")
                torchvision.utils.save_image(grid, EXPERIMENT_BASE_PATH + "/testing_images" + "/" + EXPERIMENT_NAME + "_epoch_" + str(epoch) + "_batch_" + str(batch_idx) + ".png")
                # Update the tensorboard
                writer.add_scalar('Test/Loss', loss.item()/BATCH_SIZE,
                                epoch * num_test_batches + batch_idx)
                writer.add_scalar('Test/KL Div Loss', kld_loss.item()/BATCH_SIZE,
                                    epoch * num_test_batches + batch_idx)
                              
        # Print the statistics
        print('Test Loss:', loss_meter.avg)
        loss_meter.reset()

    return model


def main():
    global USE_GUIDED_ATTENTION_LOSS
    global EXPERIMENT_NAME
    global EXPERIMENT_BASE_PATH

    # Load Dataset
    print("Loading train dataset from ", TFRECORD_FOLDER)
    train_dataset = DepthSemanticImageDataset(tfrecord_folder=TFRECORD_FOLDER, shuffle=True, batch_size=BATCH_SIZE, one_tfrecord=ONE_TFRECORD)
    print("Loading test dataset", TFRECORD_TEST_FOLDER)
    test_dataset = DepthSemanticImageDataset(tfrecord_folder=TFRECORD_TEST_FOLDER, shuffle=False, batch_size=BATCH_SIZE, one_tfrecord=ONE_TFRECORD, test_dataset=True)

    # Parse arguments from argparser 
    args = parser.parse_args()

    # Check if load model is true , and if so, get the model path
    if args.load_model:
        LOAD_MODEL = True
        model_path = args.model_path
        LOAD_MODEL_FILE = args.model_filesss
        print("Loading model from ", model_path)
    else:
        LOAD_MODEL = False
        LOAD_MODEL_FILE = None
        model_path = None
    
    EXPERIMENT_NAME = args.experiment_name
    EXPERIMENT_BASE_PATH = os.path.join(BASE_PATH, EXPERIMENT_NAME)

    # Define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1, collate_fn=collate_batch)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, collate_fn=collate_batch, shuffle=False)
    print("Loaded data loaders")

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))
    
    EXPERIMENT_BASE_PATH = os.path.join(BASE_PATH, "experiments", EXPERIMENT_NAME)
    # initialize the folder for saving model, params, config, and training and testing images
    if not os.path.exists(EXPERIMENT_BASE_PATH):
        os.makedirs(os.path.join(EXPERIMENT_BASE_PATH))
        os.makedirs(os.path.join(EXPERIMENT_BASE_PATH, "models"))
        os.makedirs(os.path.join(EXPERIMENT_BASE_PATH, "training_images"))
        os.makedirs(os.path.join(EXPERIMENT_BASE_PATH, "testing_images"))
        os.makedirs(os.path.join(EXPERIMENT_BASE_PATH, "tensorboard"))
        writer = SummaryWriter(log_dir=os.path.join(EXPERIMENT_BASE_PATH, "tensorboard"))

    # elif EXPERIMENT_NAME == "default":
    #     pass
    else:
        print("Experiment name already exists. Please choose another name.")
        sys.exit(0)

    # Define and initialize the model
    model = VAE(latent_dim=LATENT_DIM, with_logits=True)
    model = model.to(device)
    summary(model, (1, 270, 480))
    if MULTI_GPU:
        model = nn.DataParallel(model)
        model = model.to(device0)
    if LOAD_MODEL:
        if MULTI_GPU:
            model.module.load_state_dict(torch.load(LOAD_MODEL_FILE))
        else:
            model.load_state_dict(torch.load(LOAD_MODEL_FILE))

    loss = unweighted_semantic_reconstruction_with_MSE_KLD_Loss_unweighted_for_invalid_pixels

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    TF_RECORDS_TRAINING = train_dataset.tfrecord_fnames
    TF_RECORDS_TESTING = test_dataset.tfrecord_fnames


    # populate a file detailing the config as a yaml file and save them to a file
    # the details should include batch sizes, tf_record file names for training and testing, latent spaces, loss function name, etc.
    config = {
        "base_path": BASE_PATH,
        "experiment_base_path": EXPERIMENT_BASE_PATH,
        "experiment name": EXPERIMENT_NAME,
        "batch_size": BATCH_SIZE,
        "tfrecord_folder": TFRECORD_FOLDER,
        "tfrecord_test_folder": TFRECORD_TEST_FOLDER,
        "latent_dim": LATENT_DIM,
        "loss": loss.__name__,
        "loss_function_definition": inspect.getsource(loss),
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "save_interval": SAVE_INTERVAL,
        "experiment_name": EXPERIMENT_NAME,
        "load_model": LOAD_MODEL,
        "load_model_file": LOAD_MODEL_FILE,
        "model_path": model_path,
        "multi_gpu": MULTI_GPU,
        "use_filtered_data_to_fill_in_undefined_pixels": FILL_UNDEFINED_PIXELS_WITH_FILTERED_DATA,
        "use_negative_values_to_fill_in_undefined_pixels": FILL_UNDEFINED_PIXELS_WITH_NEGATIVE_VALUES,
        "tf_records_training": TF_RECORDS_TRAINING,
        "tf_records_testing": TF_RECORDS_TESTING
        }
    
    # write config to a yaml file and save it to the experiment folder
    with open(os.path.join(EXPERIMENT_BASE_PATH, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
        
    # Print the current torch seed
    print("Current seed for training: ", torch.seed())

    # Train the model
    model = train_model(model, train_loader, test_loader, NUM_EPOCHS, writer, BATCH_SIZE,
                        loss, optimizer, save_interval=SAVE_INTERVAL)


if __name__ == "__main__":
    main()
