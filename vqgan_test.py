import argparse
from vqgan import VQGAN
from utils import plot_images
from datamodules.Images import ImagesDataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension n_z",
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
        default=4.5e-06,
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
        default=30000,
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

    model = VQGAN.load_from_checkpoint("crossface-256z-16b/vqgan/last.ckpt", args=args)
    model.eval()

    data_module = ImagesDataModule(
        image_size=args.image_size, batch_size=args.batch_size, num_workers=2
    )
    data_module.setup()

    batch = next(iter(data_module.test_dataloader()))

    reconstructed_images = model.reconstruct_images(batch)

    images = {"input": batch, "reconstruction": reconstructed_images}

    plot_images(images)
