import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Match these to your training config
image_size = 64
latent_dim = 100
output_path = "./static/generated"

# Generator architecture from training time
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)




latent_dim = 100
generator = Generator(latent_dim)
generator.load_state_dict(torch.load('generator.pth', map_location='cpu'))
generator.eval()

z = torch.randn(1, latent_dim)
generated_img = generator(z)


save_image(generated_img, 'static/generated/generated.png', normalize=True)



print("✅ Image generated and saved to static/generated/generated.png")
