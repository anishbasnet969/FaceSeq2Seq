import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from metrics.fs_metrics import FaceSemanticMetrics

from vqgan import VQGAN
from transformers import RobertaModel, RobertaTokenizer
from transformer_decoder import TransformerDecoder


class CrossFace(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        self.transformer_encoder = RobertaModel.from_pretrained("roberta-large")
        embedding_dim = self.transformer_encoder.config.hidden_size

        self.transformer_decoder = TransformerDecoder(
            args.num_codebook_vectors, embedding_dim, args.block_size
        )

        self.fid = FrechetInceptionDistance(feature=2048)
        self.fs_metrics = FaceSemanticMetrics()

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        # model = VQGAN.load_from_checkpoint(args.checkpoint_path)
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

    def forward(self, texts, images):
        _, indices = self.encode_to_z(images)

        tokenized_text = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        encoder_output = self.transformer_encoder(**tokenized_text)

        sos_tokens = torch.ones(images.shape[0], 1) * self.sos_token
        input_indices = torch.cat((sos_tokens, indices), dim=1)

        logits = self.transformer_decoder(
            input_indices[:, :-1].long(), encoder_output.last_hidden_state
        )

        target = indices

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = float("-inf")
        return out

    @torch.no_grad()
    def sample(self, texts, steps, temperature=1.0, top_k=100):
        self.transformer_encoder.eval()
        self.transformer_decoder.eval()

        sos_tokens = torch.ones(len(texts), 1) * self.sos_token
        x = sos_tokens

        tokenized_text = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        encoder_output = self.transformer_encoder(**tokenized_text)

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
        texts, imgs = batch
        logits, target = self(texts, imgs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        texts, imgs = batch
        logits, target = self(texts, imgs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log(
            "val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        texts, imgs = batch

        sampled_indices = self.sample(texts, steps=256)
        generated_imgs = self.z_to_image(sampled_indices)

        resized_generated_imgs = F.interpolate(
            generated_imgs, size=(299, 299), mode="bicubic", antialias=True
        )
        resized_real_imgs = F.interpolate(
            imgs, size=(299, 299), mode="bicubic", antialias=True
        )

        self.fid.update(resized_generated_imgs, real=False)
        self.fid.update(resized_real_imgs, real=True)
        self.fs_metrics.update(resized_generated_imgs, resized_real_imgs)

    def on_test_epoch_end(self, outputs):
        fid_score = self.fid.compute()
        fsd, fss, cos_sim = self.fs_metrics.compute()
        log_dict = {
            "test/fid": fid_score.detach(),
            "test/fsd": fsd.detach(),
            "test/fss": fss.detach(),
            "test/cos_sim": cos_sim.detach(),
        }
        self.log_dict(log_dict, logger=True, on_epoch=True, prog_bar=True)

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

        param_dict = {pn: p for pn, p in self.transformer_decoder.named_parameters()}

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

        optimizer_decoder = torch.optim.AdamW(
            optim_groups, lr=4.5e-04, betas=(0.9, 0.95)
        )

        return optimizer_decoder
