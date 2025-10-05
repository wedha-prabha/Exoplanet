# src/preprocessing.py
import torch
import torch.nn as nn
import numpy as np

class DenoisingAutoencoder1D(nn.Module):
    """
    Small 1D conv autoencoder to denoise light curves.
    """
    def __init__(self, input_len=512, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_len//8)*64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (input_len//8)*64),
            nn.ReLU(),
            nn.Unflatten(1, (64, input_len//8)),
            nn.ConvTranspose1d(64,32,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32,16,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16,1,kernel_size=5,stride=2,padding=2,output_padding=1),
        )

    def forward(self, x):
        # x: (B, 1, L)
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def apply_denoise(autoencoder, x_numpy, device="cpu"):
    """
    x_numpy: (N, L)
    """
    autoencoder.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_numpy).unsqueeze(1).float().to(device)
        out = autoencoder(x).cpu().numpy().squeeze(1)
    return out

if __name__ == "__main__":
    # test
    import numpy as np
    ae = DenoisingAutoencoder1D(input_len=512)
    x = np.random.normal(size=(4,512)).astype(np.float32)
    y = apply_denoise(ae, x)
    print("denoised shape", y.shape)
