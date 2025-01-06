import torch
import torch.nn as nn
import torch.nn.functional as F


class DVAEELBOLoss(nn.Module):
    def __init__(self, kl_div_weight, codebook_size):
        super().__init__()

        self.kl_div_weight = kl_div_weight
        self.codebook_size = codebook_size


    def forward(self, x, x_hat, logits):

        log_softmax = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.ones_like(logits) / self.codebook_size)

        elbo_loss = torch.nn.functional.kl_div(
            log_softmax, log_uniform, log_target=True, reduction="batchmean"
        )
        
        mse_loss = F.mse_loss(x_hat, x)
        loss = mse_loss + self.kl_div_weight * elbo_loss

        return loss