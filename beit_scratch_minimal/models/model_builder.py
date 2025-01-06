import torch
import torch.nn as nn
import torch.nn.functional as F


class DVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        temperature: float,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.codebook = nn.Embedding(
            self.codebook_size, self.codebook_dim
        )

    def encode(self, x):
        logits = self.encoder(x).permute(0, 2, 3, 1)
        
        soft_logits = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        
        codebook_enc = torch.matmul(soft_logits, self.codebook.weight)
        
        return codebook_enc, logits
    
    def forward(self, x):
        codebook_enc, logits = self.encode(x)
        x_hat = self.decoder(codebook_enc.permute(0, 3, 1, 2))
        return x_hat, logits


def build_dvae_model(
        encoder: nn.Module,
        decoder: nn.Module,
        temperature: float,
        codebook_size: int,
        codebook_dim: int):
    
    return DVAE(encoder, decoder, temperature, codebook_size, codebook_dim)