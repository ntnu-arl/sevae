import torch
import torch.nn as nn

def unweighted_non_semantic_reconstruction_with_MSE_KLD_Loss_unweighted_for_invalid_pixels(x, x_logit, semantic_image, gcam_map, mean, logvar, beta_coeff=1.0, latent_dims=128):
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(torch.sigmoid(x_logit), x) * invalid_pixel_mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_norm = (beta_coeff*latent_dims)/(480.0*270.0)
    kld_loss = kld_loss * beta_norm
    return reconstruction_loss + kld_loss, kld_loss


def unweighted_semantic_reconstruction_with_MSE_KLD_Loss_unweighted_for_invalid_pixels(x, x_logit, semantic_image, gcam_map, mean, logvar):
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")

    # Get unique semantic labels
    unique_semantic_labels, unique_counts = torch.unique(semantic_image[semantic_image > 10], return_counts=True)
    weight_matrix = torch.ones_like(semantic_image)

    # for each unique semantic label, get the number of pixels in the semantic mask and mask the invalid pixels
    for i in range(len(unique_semantic_labels)):
        if unique_semantic_labels[i] < 10:
            continue
        counts = unique_counts[i]
        weight_per_pixel = 1.0
        if counts > 40:
            weight_per_pixel = max(6000.0 / counts, 15.0)
        
        # add weight to weight matrix based on the number of pixels per semantic label
        weight_matrix[semantic_image == unique_semantic_labels[i]] *= weight_per_pixel
    cross_ent = MSE_LOSS(torch.sigmoid(x_logit), x) * invalid_pixel_mask * weight_matrix
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_norm = (1.0*128)/(480.0*270.0)
    kld_loss = kld_loss * beta_norm
    return reconstruction_loss + kld_loss, kld_loss

def vanilla_image_MSE_Loss(x, x_hat):
    invalid_pixel_mask = torch.where(x_hat > 0, torch.ones_like(x_hat), torch.zeros_like(x_hat))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(x_hat, x)*invalid_pixel_mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    return reconstruction_loss