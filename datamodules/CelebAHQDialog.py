import os
import json
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ImageTextPaths(Dataset):
    def __init__(self, paths, captions_file, size=None):
        self.size = size

        self.paths = paths
        self._length = len(self.paths)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        with open(captions_file, "r") as json_file:
            self.captions_data = json.load(json_file)

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
        image_path = self.paths[i]
        image = self.preprocess_image(image_path)
        text = self.captions_data[os.path.basename(image_path)]["overall_caption"]
        return text, image


class CelebAHQDialogTrain(Dataset):
    def __init__(self, image_size):
        super().__init__()
        root = "dataset\image"
        captions_file = "dataset\captions_hq.json"
        with open("data\train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImageTextPaths(
            paths=paths, captions_file=captions_file, size=image_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class CelebAHQDialogValidation(Dataset):
    def __init__(self, image_size):
        super().__init__()
        root = "dataset\image"
        captions_file = "dataset\captions_hq.json"
        with open("data\validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImageTextPaths(paths=paths, captions_file=captions_file, size=image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CelebAHQDialogDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, image_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage):
        self.train_dataset = CelebAHQDialogTrain(image_size=self.image_size)
        self.val_dataset = CelebAHQDialogValidation(image_size=self.image_size)

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
