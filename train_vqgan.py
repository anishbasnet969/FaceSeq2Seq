import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import XLAStrategy

from vqgan import VQGAN
from datamodules.CelebAHQImages import CelebAHQImagesDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Latent dimension n_z (for compatibility with roberta large hidden dim: 1024).",
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
        "--batch-size",
        type=int,
        default=4,
        help="Input batch size for training (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4.5e-04,
        help="Learning rate",
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
    parser.add_argument(
        "--disc-factor", type=float, default=1.0, help="Discriminator weighing factor"
    )
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

    logger = TensorBoardLogger("crossface_logs", name="vqgan")

    checkpoint_callback = ModelCheckpoint(
        dirpath="vqgan_checkpoints/",
        filename="vqgan-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=50,
    )

    vqgan = VQGAN(args)

    data_module = CelebAHQImagesDataModule(
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

    trainer.fit(vqgan, datamodule=data_module)
    # trainer.fit(vqgan, datamodule=data_module, ckpt_path="path_to_my_checkpoint.ckpt")