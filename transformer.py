import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vqgan import VQGAN
from transformers import BertModel, BertTokenizer


class FaceSeq2Seq(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        embedding_dim = self.bert_encoder.config.hidden_size

        self.target_embedding = nn.Embedding(
            num_embeddings=args.num_codebook_vectors, embedding_dim=embedding_dim
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=12)

        self.head = nn.Linear(embedding_dim, args.num_codebook_vectors)

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

    def forward(self, input_ids, attention_mask, images):
        _, indices = self.encode_to_z(images)

        encoder_output = self.bert_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sos_tokens = torch.ones(images.shape[0], 1) * self.sos_token
        input_indices = torch.cat((sos_tokens, indices), dim=1)

        input_indices_embeddings = self.target_embedding(input_indices[:, :-1])

        seq_length = indices.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length)

        decoder_output = self.decoder(
            tgt=input_indices_embeddings,
            memory=encoder_output.last_hidden_state,
            tgt_mask=tgt_mask,
        )

        logits = self.head(decoder_output)

        target = indices

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = float("-inf")
        return out
