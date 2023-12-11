import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ImageTextPaths(Dataset):
    def __init__(self, paths, captions_folder, size=None, captions_per_image=10):
        self.size = size
        self.paths = paths
        self.captions_folder = captions_folder
        self.captions_per_image = captions_per_image
        self._length = len(self.paths) * captions_per_image

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
    
    def load_captions(self, image_path):
        basename = os.path.basename(image_path)
        caption_filename = os.path.splitext(basename)[0] + '.txt'
        caption_path = os.path.join(self.captions_folder, caption_filename)
        with open(caption_path, 'r') as file:
            captions = file.read().splitlines()
        return captions

    def __getitem__(self, i):
        image_index = i // self.captions_per_image
        caption_index = i % self.captions_per_image

        image_path = self.paths[image_index]
        image = self.preprocess_image(image_path)
        captions = self.load_captions(image_path)

        return captions[caption_index], image


class MMCelebAHQTrain(Dataset):
    def __init__(self, image_size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        captions_folder = "MultiModalCelebAHQ/dataset/captions/celeba-caption"
        with open(r"data/train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImageTextPaths(
            paths=paths, captions_folder=captions_folder, size=image_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class MMCelebAHQValidation(Dataset):
    def __init__(self, image_size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        captions_folder = "MultiModalCelebAHQ/dataset/captions/celeba-caption"
        with open(r"data/validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImageTextPaths(
            paths=paths, captions_folder=captions_folder, size=image_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class MMCelebAHQValidation(Dataset):
    def __init__(self, image_size):
        super().__init__()
        root = "MultiModalCelebAHQ/dataset/images"
        captions_folder = "MultiModalCelebAHQ/dataset/captions/celeba-caption"
        with open(r"data/test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImageTextPaths(
            paths=paths, captions_folder=captions_folder, size=image_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]