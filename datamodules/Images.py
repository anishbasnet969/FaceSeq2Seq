import os
import numpy as np
import albumentations
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ImagePaths(Dataset):
    def __init__(self, paths, size=None):
        self.size = size

        self.paths = paths
        self._length = len(self.paths)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.paths[i])
        return example


class ImagesTrain(Dataset):
    def __init__(self, size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        with open(r"data/train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class ImagesValidation(Dataset):
    def __init__(self, size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        with open(r"data/validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example
    
class ImagesTest(Dataset):
    def __init__(self, size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        with open(r"data/test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class ImagesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, image_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        self.train_dataset = ImagesTrain(size=self.image_size)
        self.val_dataset = ImagesValidation(size=self.image_size)
        self.test_dataset = ImagesTest(size=self.image_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
