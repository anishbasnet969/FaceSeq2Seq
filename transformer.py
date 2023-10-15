import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vqgan import VQGAN
from transformers import BertModel, BertTokenizer
from transformer_decoder import TransformerDecoder

from datamodules.img_txt import CelebAHQImageTextDataModule


class FaceSeq2Seq(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        self.transformer_encoder = BertModel.from_pretrained("bert-base-uncased")
        embedding_dim = self.transformer_encoder.config.hidden_size

        self.transformer_decoder = TransformerDecoder(
            args.num_codebook_vectors, embedding_dim, args.block_size
        )

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(
            indices.shape[0], p1, p2, 768
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        images = self.vqgan.decode(ix_to_vectors)
        return images

    def forward(self, tokenized_text, images):
        _, indices = self.encode_to_z(images)

        encoder_output = self.transformer_encoder(tokenized_text)

        sos_tokens = torch.ones(images.shape[0], 1) * self.sos_token
        input_indices = torch.cat((sos_tokens, indices), dim=1)

        logits = self.transformer_decoder(
            input_indices[:, :-1], encoder_output.last_hidden_state
        )

        target = indices

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = float("-inf")
        return out

    @torch.no_grad()
    def sample(self, tokenized_text, x, steps, temperature=1.0, top_k=100):
        self.transformer_encoder.eval()
        self.transformer_decoder.eval()

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        x = torch.cat((sos_tokens, x), dim=1)

        encoder_output = self.transformer_encoder(tokenized_text)

        for k in range(steps):
            logits = self.transformer_decoder(x, encoder_output.last_hidden_state)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, 1:]
        self.transformer_encoder.train()
        self.transformer_encoder.train()
        return x

    def training_step(self, batch, batch_idx):
        tokenized_text, imgs = batch
        logits, target = self(tokenized_text, imgs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        tokenized_text, imgs = batch
        logits, target = self(tokenized_text, imgs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log(
            "val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.transformer_decoder.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceSeq2Seq")
    parser.add_argument(
        "--latent-dim", type=int, default=768, help="Latent dimension n_z."
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image height and width.)"
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.25, help="Commitment loss scalar."
    )
    parser.add_argument(
        "--image-channels", type=int, default=3, help="Number of channels of images."
    )
    parser.add_argument(
        "--dataset-path", type=str, default="./data", help="Path to data."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/last_ckpt.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device the training is on"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Input batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.25e-05, help="Learning rate."
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta param.")
    parser.add_argument("--beta2", type=float, default=0.9, help="Adam beta param.")
    parser.add_argument(
        "--disc-start", type=int, default=10000, help="When to start the discriminator."
    )
    parser.add_argument(
        "--disc-factor",
        type=float,
        default=1.0,
        help="Weighting factor for the Discriminator.",
    )
    parser.add_argument(
        "--l2-loss-factor",
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

    parser.add_argument(
        "--sos-token", type=int, default=0, help="Start of Sentence token."
    )

    args = parser.parse_args()

    faceseq2seq = FaceSeq2Seq(args)

    data_module = CelebAHQImageTextDataModule(
        image_size=args.image_size, batch_size=args.batch_size, num_workers=2
    )

    trainer = pl.Trainer()

    trainer.fit(faceseq2seq, data_module)
