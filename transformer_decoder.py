import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_size))

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=12)

        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, indices, memory):
        token_embeddings = self.tok_emb(indices)
        seq_length = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :seq_length, :]
        input_embeddings = token_embeddings + position_embeddings

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length)

        x = self.decoder(
            tgt=input_embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
