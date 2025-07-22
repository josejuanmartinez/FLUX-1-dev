import zipfile
import tempfile

from tqdm.auto import tqdm

from torch.utils.data import Dataset
from torchvision import transforms
import hashlib

from transformers import T5EncoderModel

from PIL import Image, ImageOps
import torch
import os

from diffusers import FluxPipeline


class ZipDataset(Dataset):
    """
    Dataset for loading image-text pairs from a ZIP file.
    Each image should have a corresponding text file with the same base name.
    The text file should contain the prompt for the image.
    The dataset applies transformations to the images and computes prompt embeddings using a specified model.
    """
    def __init__(self, zip_path, width, height, max_sequence_length, model_id, trigger_word, accelerator_device):
        self.zip_path = zip_path
        self.width = width
        self.height = height
        self.max_sequence_length = max_sequence_length
        self.trigger_word = trigger_word
        self.device = accelerator_device

        # Extract ZIP
        self.temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.temp_dir.name)

        # Load image-text pairs
        self.instance_images = []
        self.prompts = []
        self.image_hashes = []

        for fname in os.listdir(self.temp_dir.name):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(fname)[0]
                image_path = os.path.join(self.temp_dir.name, fname)
                text_path = os.path.join(self.temp_dir.name, f"{base_name}.txt")

                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                    image = Image.open(image_path).convert("RGB")
                    self.instance_images.append(image)
                    self.prompts.append(prompt)
                    self.image_hashes.append(self._hash_image(image))

        if not self.instance_images:
            raise ValueError("No valid image-text pairs found.")

        # Preprocess images
        self.pixel_values = self._apply_transforms()

        # Compute prompt embeddings
        self.data_dict = self._compute_prompt_embeddings(model_id)
        self._length = len(self.instance_images)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = index % self._length
        hash_key = self.image_hashes[idx]
        prompt_embeds, pooled_prompt_embeds, text_ids = self.data_dict[hash_key]
        return {
            "instance_images": self.pixel_values[idx],
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    def _hash_image(self, image):
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _apply_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return [transform(self.exif_transpose(img)) for img in self.instance_images]

    def _load_flux_pipeline(self, model_id):
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )

        pipeline = FluxPipeline.from_pretrained(
            model_id,
            text_encoder_2=text_encoder,
            transformer=None,
            vae=None,
            torch_dtype=torch.float16
        )

        pipeline.to(self.device)
        return pipeline

    @torch.no_grad()
    def _compute_prompt_embeddings(self, model_id):
        pipeline = self._load_flux_pipeline(model_id)
        data_dict = {}

        for i, prompt in enumerate(tqdm(self.prompts, desc=f"Encoding prompts with FLUX. Trigger word: {self.trigger_word}")):
            full_prompt = f"{self.trigger_word} {prompt}"
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=full_prompt,
                prompt_2=None,
                max_sequence_length=self.max_sequence_length,
            )

            data_dict[self.image_hashes[i]] = (
                prompt_embeds.squeeze(0).cpu(),
                pooled_prompt_embeds.squeeze(0).cpu(),
                text_ids.squeeze(0).cpu()
            )

        return data_dict

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([ex["instance_images"] for ex in examples]).float()
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format)
        prompt_embeds = torch.stack([ex["prompt_embeds"] for ex in examples])
        pooled_prompt_embeds = torch.stack([ex["pooled_prompt_embeds"] for ex in examples])
        text_ids = torch.stack([ex["text_ids"] for ex in examples])[0]  # keep one copy

        return {
            "pixel_values": pixel_values,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    @staticmethod
    def exif_transpose(image):
        return ImageOps.exif_transpose(image)