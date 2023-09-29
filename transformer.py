import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from seq_map import Seq2SeqTransformer


class FaceSeq2Seq(pl.LightningModule):
    def __init__(
        self,
        transformer_config,
        first_stage_config,
        ckpt_path=None,
        ignore_keys=[],
        first_stage_key="image",
        pkeep=1.0,
        sos_token=0,
    ):
        super().__init__()
