import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import XLAStrategy

from models.cf_transformer import CrossFace
from datamodules.img_txt import CelebAHQImageTextDataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossFace")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Latent dimension n_z (for compatibility with roberta large hidden dim: 1024).",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image height and width."
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors.",
    )
    parser.add_argument(
        "--block-size", type=int, default=256, help="Transformer Decoder Block Size"
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
        "--learning-rate", type=float, default=4.5e-04, help="Learning rate."
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

    logger = TensorBoardLogger("crossface_logs", name="cf_transformer")

    checkpoint_callback = ModelCheckpoint(
        dirpath="cf_transformer_checkpoints/",
        filename="cf_transformer-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=50,
    )

    crossface = CrossFace(args)

    data_module = CelebAHQImageTextDataModule(
        image_size=args.image_size, batch_size=args.batch_size, num_workers=2
    )

    trainer = pl.Trainer(
        logger=logger,
        strategy=XLAStrategy(),
        devices=32,
        callbacks=[checkpoint_callback],
        max_epochs=500,
        precision=16,
    )

    trainer.fit(crossface, data_module)
    # trainer.fit(crossface, datamodule=data_module, ckpt_path="path_to_my_checkpoint.ckpt")
