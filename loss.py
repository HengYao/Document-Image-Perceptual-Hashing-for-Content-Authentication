import torch
import torch.nn.functional as F


def perceptual_hash_loss(hash_code):
    if hash_code.ndim != 2:
        raise ValueError("hash_code must have shape [71, hash_length]")
    if hash_code.size(0) != 71:
        raise ValueError("hash_code must contain 71 samples")

    anchor = hash_code[0]
    similar_loss_sum = 0
    different_loss_sum = 0

    for index in range(1, 36):
        similar_loss_sum = similar_loss_sum + torch.sigmoid(
            F.mse_loss(anchor, hash_code[index])
        )

    for index in range(36, 71):
        different_loss_sum = different_loss_sum + torch.sigmoid(
            F.mse_loss(anchor, hash_code[index])
        )

    similar_loss = similar_loss_sum / 35
    different_loss = different_loss_sum / 35
    loss = similar_loss - different_loss

    return {
        "similar_loss": similar_loss,
        "different_loss": different_loss,
        "loss": loss,
    }
