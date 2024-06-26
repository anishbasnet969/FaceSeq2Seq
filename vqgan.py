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

        self.automatic_optimization = False

        self.args = args

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

    def calculate_lambda(self, nll_loss, g_loss):
        try:
            last_layer = self.decoder.model[-1]
            last_layer_weight = last_layer.weight

            nll_loss_grads = torch.autograd.grad(
                nll_loss, last_layer_weight, retain_graph=True
            )[0]
            g_loss_grads = torch.autograd.grad(
                g_loss, last_layer_weight, retain_graph=True
            )[0]

            λ = torch.norm(nll_loss_grads) / (torch.norm(g_loss_grads) + 1e-4)
            λ = torch.clamp(λ, 0, 1e4).detach()
        except RuntimeError:
            assert not self.training
            λ = torch.tensor(0.0)
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.0):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def training_step(self, batch, batch_idx):
        imgs = batch
        decoded_images, _, q_loss = self(imgs)

        opt_vq, opt_disc = self.optimizers()

        logits_real = self.discriminator(imgs)
        logits_fake = self.discriminator(decoded_images)

        disc_factor = self.adopt_weight(
            self.args.disc_factor, self.global_step, threshold=self.args.disc_start
        )

        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.args.perceptual_loss_factor * perceptual_loss
            + self.args.rec_loss_factor * rec_loss
        )
        nll_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(logits_fake)

        λ = self.calculate_lambda(nll_loss, g_loss)
        vq_loss = nll_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - logits_real))
        d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        disc_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        self.log(
            "train/vqloss",
            vq_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/discloss",
            disc_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        log_dict = {
            "train/vq_loss": vq_loss.clone().detach().mean(),
            "train/q_loss": q_loss.detach().mean(),
            "train/nll_loss": nll_loss.detach().mean(),
            "train/perceptual_rec_loss": perceptual_rec_loss.detach().mean(),
            "train/rec_loss": rec_loss.detach().mean(),
            "train/perceptual_loss": perceptual_loss.detach().mean(),
            "train/disc_weight": λ.detach(),
            "train/disc_factor": torch.tensor(disc_factor),
            "train/g_loss": g_loss.detach().mean(),
            "train/disc_loss": disc_loss.clone().detach().mean(),
            "train/logits_real": logits_real.detach().mean(),
            "train/logits_fake": logits_fake.detach().mean(),
        }
        self.log_dict(
            log_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)

        opt_disc.zero_grad()
        disc_loss.backward()

        opt_vq.step()
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        imgs = batch
        decoded_images, _, q_loss = self(imgs)

        logits_real = self.discriminator(imgs)
        logits_fake = self.discriminator(decoded_images)

        disc_factor = self.adopt_weight(
            self.args.disc_factor, self.global_step, threshold=self.args.disc_start
        )

        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.args.perceptual_loss_factor * perceptual_loss
            + self.args.rec_loss_factor * rec_loss
        )
        nll_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(logits_fake)

        λ = self.calculate_lambda(nll_loss, g_loss)
        vq_loss = nll_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - logits_real))
        d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        disc_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        self.log(
            "val/perceptual_rec_loss",
            perceptual_rec_loss.detach().mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/vqloss",
            vq_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        log_dict = {
            "val/vq_loss": vq_loss.clone().detach().mean(),
            "val/q_loss": q_loss.detach().mean(),
            "val/nll_loss": nll_loss.detach().mean(),
            "val/rec_loss": rec_loss.detach().mean(),
            "val/perceptual_loss": perceptual_loss.detach().mean(),
            "val/disc_weight": λ.detach(),
            "val/disc_factor": torch.tensor(disc_factor),
            "val/g_loss": g_loss.detach().mean(),
            "val/disc_loss": disc_loss.clone().detach().mean(),
            "val/logits_real": logits_real.detach().mean(),
            "val/logits_fake": logits_fake.detach().mean(),
        }
        self.log_dict(log_dict)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.args.learning_rate

        opt_vq = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.codebook.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(self.args.beta1, self.args.beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(self.args.beta1, self.args.beta2),
        )

        return [opt_vq, opt_disc]

    def reconstruct_images(self, imgs):
        with torch.no_grad():
            encoded_images = self.encoder(imgs)
            quant_conv_encoded_images = self.quant_conv(encoded_images)
            codebook_mapping, _, _ = self.codebook(quant_conv_encoded_images)
            post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
            decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images
