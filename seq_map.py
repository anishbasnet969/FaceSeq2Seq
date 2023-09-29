import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Seq2SeqTransformer(nn.Module):
    def __init__(self, bert_model_name, decoder_config):
        super(Seq2SeqTransformer, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert_encoder.config.hidden_size

        self.target_embedding = nn.Embedding(
            num_embeddings=decoder_config["vocab_size"], embedding_dim=hidden_size
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, **decoder_config
        )

        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=decoder_config["num_layers"]
        )

    def forward(self, input_ids, attention_mask, decoder_input):
        bert_output = self.bert_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        embedded_decoder_input = self.target_embedding(decoder_input)

        decoder_output = self.decoder(
            tgt=embedded_decoder_input,
            memory=bert_output.last_hidden_state,
        )

        return decoder_output


decoder_config = {
    "num_layers": 6,
    "nhead": 8,
    "dim_feedforward": 2048,
    "dropout": 0.1,
}
