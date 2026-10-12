import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from setup import device, frames, env
from setup import PAPER, TEAL, GOLD, CLAY
Z = 16

class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(                            # in: 3 × 24 × 24
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),   # 16 × 12 × 12
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # 32 × 6 × 6
            nn.Flatten())                                    # 1152
        self.mu, self.logvar = nn.Linear(1152, Z), nn.Linear(1152, Z)
        self.fc = nn.Linear(Z, 1152)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid())

    def encode(self, x):                                     # x: (B, 24, 24, 3)
        h = self.enc(x.permute(0, 3, 1, 2))
        return self.mu(h), self.logvar(h)

    def decode(self, z):                                     # -> (B, 24, 24, 3)
        return self.dec(self.fc(z).view(-1, 32, 6, 6)).permute(0, 2, 3, 1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterise
        return self.decode(z), mu, logvar

vae = ConvVAE().to(device)
opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
flat = torch.tensor(frames.reshape(-1, 24, 24, 3))
track_t = torch.tensor(env.track).to(device)                 # the empty track
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(flat),
                                     batch_size=256, shuffle=True)
t0 = time.time()
for epoch in range(30):
    tot = 0.0
    for (x,) in loader:
        x = x.to(device)
        mu, logvar = vae.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterise
        z = z + 0.10 * torch.randn_like(z)   # extra fuzz: M's predictions will land
                                             # NEAR codes, so train the decoder there too
        xhat = vae.decode(z)
        w = 1.0 + 30.0 * (x - track_t).abs().mean(-1, keepdim=True)  # weight what moves
        recon = (w * F.binary_cross_entropy(xhat, x, reduction="none")).sum() / len(x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        loss = recon + 1.0 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x)
    if (epoch + 1) % 3 == 0:
        print(f"epoch {epoch+1:2d}  loss {tot/len(flat):8.2f}")
print(f"V trained in {time.time()-t0:.0f}s   "
      f"({sum(p.numel() for p in vae.parameters()):,} parameters)")





vae.eval()
with torch.no_grad():
    test = flat[:20000:2611][:8].to(device)
    recon, _, _ = vae(test)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for i in range(8):
        axes[0, i].imshow(test[i].cpu()); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().clamp(0, 1)); axes[1, i].axis("off")
    axes[0, 0].set_title("original", loc="left", color=TEAL)
    axes[1, 0].set_title("V's reconstruction (from 16 numbers)", loc="left", color=CLAY)
    fig.suptitle("V — 1,728 pixels → 16 numbers → 1,728 pixels", y=1.02)
    plt.tight_layout(); plt.savefig("plot_v_recon.png", dpi=150, bbox_inches="tight"); plt.show()

    # latent walk between two frames of the SAME episode, half a lap apart
    e_walk = 2
    z1, _ = vae.encode(torch.tensor(frames[e_walk, 0:1]).to(device))
    z2, _ = vae.encode(torch.tensor(frames[e_walk, 26:27]).to(device))
    steps = torch.linspace(0, 1, 9, device=device).view(-1, 1)
    walk = vae.decode(z1 + steps * (z2 - z1))

    fig, axes = plt.subplots(1, 9, figsize=(12, 1.8))
    for i in range(9):
        axes[i].imshow(walk[i].cpu().clamp(0, 1)); axes[i].axis("off")
    axes[0].set_title("code A", loc="left", color=TEAL)
    axes[-1].set_title("code B", loc="right", color=GOLD)
    fig.suptitle("The smooth walk — the car glides around the track between two codes", y=1.15)
    plt.tight_layout(); plt.savefig("plot_v_walk.png", dpi=150, bbox_inches="tight"); plt.show()