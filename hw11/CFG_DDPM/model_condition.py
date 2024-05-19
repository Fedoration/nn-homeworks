import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules_condition import UNetConditional
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, c_in=1, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, c_in, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNetConditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

            plot_images(sampled_images)
            save_images(
                sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")
            )
            torch.save(
                model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt")
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join("models", args.run_name, f"optim.pt"),
            )


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 128
    args.image_size = 32
    args.num_classes = 10
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
    # save_images(x, "test.jpg")


# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import math
# from modules import *


# class DiffusionModel(pl.LightningModule):
#     def __init__(self, in_size, t_range, img_depth, t_dim=128, num_classes=None):
#         super().__init__()
#         self.beta_small = 1e-4
#         self.beta_large = 0.02
#         self.t_range = t_range
#         self.in_size = in_size
#         self.t_dim = t_dim
#         self.num_classes = num_classes

#         bilinear = True
#         self.inc = DoubleConv(img_depth, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         factor = 2 if bilinear else 1
#         self.down3 = Down(256, 512 // factor)
#         self.up1 = Up(512, 256 // factor, bilinear)
#         self.up2 = Up(256, 128 // factor, bilinear)
#         self.up3 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, img_depth)
#         self.sa1 = SAWrapper(256, 8)
#         self.sa2 = SAWrapper(256, 4)
#         self.sa3 = SAWrapper(128, 8)

#         if num_classes:
#             self.cond_emb = nn.Embedding(num_classes, t_dim)

#     def pos_encoding(self, t, channels, embed_size):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         # return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
#         return pos_enc

#     def expand_encoding(self, pos_enc, channels, embed_size):
#         return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

#     def forward(self, x, t, y):
#         """
#         Model is U-Net with added positional encodings and self-attention layers.
#         """
#         t = self.pos_encoding

#         x1 = self.inc(x)
#         x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
#         x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
#         x3 = self.sa1(x3)
#         x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
#         x4 = self.sa2(x4)
#         x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
#         x = self.sa3(x)
#         x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
#         x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
#         output = self.outc(x)
#         return output

#     def beta(self, t):
#         return self.beta_small + (t / self.t_range) * (
#             self.beta_large - self.beta_small
#         )

#     def alpha(self, t):
#         return 1 - self.beta(t)

#     def alpha_bar(self, t):
#         return math.prod([self.alpha(j) for j in range(t)])

#     def get_loss(self, batch, batch_idx):
#         """
#         Corresponds to Algorithm 1 from (Ho et al., 2020).
#         """
#         ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
#         noise_imgs = []
#         epsilons = torch.randn(batch.shape, device=self.device)
#         for i in range(len(ts)):
#             a_hat = self.alpha_bar(ts[i])
#             noise_imgs.append(
#                 (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
#             )
#         noise_imgs = torch.stack(noise_imgs, dim=0)
#         e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
#         loss = nn.functional.mse_loss(
#             e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
#         )
#         return loss

#     def denoise_sample(self, x, t):
#         """
#         Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
#         """
#         with torch.no_grad():
#             if t > 1:
#                 z = torch.randn(x.shape)
#             else:
#                 z = 0
#             e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))

#             x = x.to("cpu")
#             t = t.to("cpu")
#             e_hat = e_hat.to("cpu")

#             pre_scale = 1 / math.sqrt(self.alpha(t))
#             e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
#             post_sigma = math.sqrt(self.beta(t)) * z

#             x = pre_scale * (x - e_scale * e_hat) + post_sigma
#             return x

#     def training_step(self, batch, batch_idx):
#         loss = self.get_loss(batch, batch_idx)
#         self.log("train/loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.get_loss(batch, batch_idx)
#         self.log("val/loss", loss)
#         return

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
#         return optimizer
