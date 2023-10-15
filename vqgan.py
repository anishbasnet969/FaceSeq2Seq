import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
from discriminator import Discriminator
from losses.lpips import LPIPS
from utils import weights_init
from datamodules.img import CelebAHQImageDataModule


class VQGAN(pl.LightningModule):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.codebook = Codebook(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.discriminator = Discriminator(args)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval()

        self.prepare_training()

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quant_conv_encoded_images
        )
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quant_conv_encoded_images
        )
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.0):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch[0]
        decoded_images, _, q_loss = self(imgs)

        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            args.perceptual_loss_factor * perceptual_loss
            + args.rec_loss_factor * rec_loss
        )
        perceptual_rec_loss = perceptual_rec_loss.mean()

        if optimizer_idx == 0:
            logits_fake = self.discriminator(decoded_images)
            g_loss = -torch.mean(logits_fake)

            λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            disc_factor = self.vqgan.adopt_weight(
                args.disc_factor, self.global_step, threshold=args.disc_start
            )
            vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

            self.log(
                "train/vqloss",
                vq_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            log_dict_vq = {
                "train/total_loss": vq_loss.clone().detach().mean(),
                "train/q_loss": q_loss.detach().mean(),
                "train/perceptual_rec_loss": perceptual_rec_loss.detach().mean(),
                "train/rec_loss": rec_loss.detach().mean(),
                "train/perceptual_loss": perceptual_loss.detach().mean(),
                "train/disc_weight": λ.detach(),
                "train/disc_factor": disc_factor.detach(),
                "train/g_loss": g_loss.detach().mean(),
            }
            self.log_dict(
                log_dict_vq, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return vq_loss

        if optimizer_idx == 1:
            logits_real = self.discriminator(imgs)
            logits_fake = self.discriminator(decoded_images)

            disc_factor = self.vqgan.adopt_weight(
                args.disc_factor, self.global_step, threshold=args.disc_start
            )
            d_loss_real = torch.mean(F.relu(1.0 - logits_real))
            d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            disc_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            self.log(
                "train/discloss",
                disc_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            log_dict_disc = {
                "train/disc_loss": disc_loss.clone().detach().mean(),
                "train/logits_real": logits_real.detach().mean(),
                "train/logits_fake": logits_fake.detach().mean(),
            }
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )

            return disc_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        decoded_images, _, q_loss = self(imgs)

        logits_real = self.discriminator(imgs)
        logits_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(
            args.disc_factor, self.global_step, threshold=args.disc_start
        )

        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            args.perceptual_loss_factor * perceptual_loss
            + args.rec_loss_factor * rec_loss
        )
        perceptual_rec_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(logits_fake)

        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - logits_real))
        d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        disc_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        self.log(
            "val/perceptual_rec_loss",
            perceptual_rec_loss.detach().mean(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/vqloss",
            vq_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        log_dict_vq = {
            "val/total_loss": vq_loss.clone().detach().mean(),
            "val/q_loss": q_loss.detach().mean(),
            "val/perceptual_rec_loss": perceptual_rec_loss.detach().mean(),
            "val/rec_loss": rec_loss.detach().mean(),
            "val/perceptual_loss": perceptual_loss.detach().mean(),
            "val/disc_weight": λ.detach(),
            "val/disc_factor": disc_factor.detach(),
            "val/g_loss": g_loss.detach().mean(),
        }
        log_dict_disc = {
            "train/disc_loss": disc_loss.clone().detach().mean(),
            "train/logits_real": logits_real.detach().mean(),
            "train/logits_fake": logits_fake.detach().mean(),
        }
        self.log(log_dict_vq)
        self.log(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self, args):
        lr = args.learning_rate

        opt_vq = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(args.beta1, args.beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(args.beta1, args.beta2)
        )

        return [opt_vq, opt_disc], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=768,
        help="Latent dimension n_z (for compatibility with bert hidden dim: 768)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image height and width (default: 256)",
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Commitment loss scalar (default: 0.25)",
    )
    parser.add_argument(
        "--image-channels",
        type=int,
        default=3,
        help="Number of channels of images (default: 3)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data",
        help="Path to data (default: /data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Input batch size for training (default: 6)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.25e-05,
        help="Learning rate (default: 0.0002)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Adam beta param (default: 0.0)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.9, help="Adam beta param (default: 0.999)"
    )
    parser.add_argument(
        "--disc-start",
        type=int,
        default=10000,
        help="When to start the discriminator (default: 0)",
    )
    parser.add_argument("--disc-factor", type=float, default=1.0, help="")
    parser.add_argument(
        "--rec-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for reconstruction loss.",
    )
    parser.add_argument(
        "--perceptual-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for perceptual loss.",
    )

    args = parser.parse_args()

    vqgan = VQGAN(args)

    data_module = CelebAHQImageDataModule(
        image_size=args.image_size, batch_size=args.batch_size, num_workers=2
    )

    trainer = pl.Trainer()

    trainer.fit(vqgan, data_module)
