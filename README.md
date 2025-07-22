---
title: FLUX1-Dev Developer hub
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
short_description: Trains LORA adapters on top of FLUX1-dev
python_version: 3.10
sdk_version: 3.50.2
suggested_hardware: l4x4
suggested_storage: small
---

# 🚀 FLUX-1 Developer Hub

A powerful platform for training and using LoRA adapters with the [FLUX-1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) image generation model. This interface provides a user-friendly Gradio UI for:

* 🔍 Inference with the base FLUX-1 model
* 🧪 Fine-tuning new LoRA adapters on your custom dataset
* 🧠 Inference with your trained adapters
* ⚡ Multi-GPU accelerated training with memory-efficient optimizations

---

## 🌐 Hosted Application

This app is deployed as a **private Hugging Face Space**:

🔒 **[https://huggingface.co/spaces/jjmcarrascosa/FLUX1-dev](https://huggingface.co/spaces/jjmcarrascosa/FLUX1-dev)**

> 📩 Access is restricted. Please contact the maintainer to request permission.

---

## 📁 Dataset Format

The training dataset should be a ZIP archive with image-prompt pairs:

```
my_dataset.zip
├── 0001.png
├── 0001.txt
├── 0002.png
├── 0002.txt
...
```

Each `.txt` file should contain a short text description corresponding to the image.

---

## 🎓 Training a LoRA Adapter

From the **"🧪 Train Adapter"** tab:

1. Upload your dataset ZIP.
2. Choose a **trigger word** (e.g., `<retro>`).
3. Customize training options (steps, batch size, learning rate).
4. Enable or disable quantization and memory optimizations.
5. Click **🎯 Train Adapter**.

Training will run in a subprocess using `accelerate launch`, and you’ll see real-time progress and logs in the interface.

LoRA weights are saved to `lora_output/`.

---

## 🎨 Inference

### 🟡 Foundational Inference

Use the base FLUX-1 model directly, no adapter needed.

### 🟢 Adapted Inference

Use your fine-tuned adapter by including the trigger word in the prompt.

---

## 🧪 Advanced Training Options

| Option                        | Description                                  |
| ----------------------------- | -------------------------------------------- |
| **LoRA Rank**                 | Controls adapter capacity (4–16 is typical). |
| **8-bit Adam**                | Saves memory with minimal quality loss.      |
| **Gradient Checkpointing**    | Slower training, prevents OOM errors.        |
| **Quantization**              | Trains the transformer in 4-bit NF4 (QLoRA). |
| **Cache Latents**             | Improves speed, increases disk use.          |
| **Batch Size / Accumulation** | Larger batches = more stable learning.       |

---

## 🧩 Environment Variables

| Name                 | Required | Description                                                      |
| -------------------- | -------- | ---------------------------------------------------------------- |
| `FOUNDATIONAL_MODEL` | ✅        | Model ID from Hugging Face (e.g. `black-forest-labs/FLUX.1-dev`) |

---

## 🧠 Internals

* **Transformer-only fine-tuning** via LoRA
* **T5 text encoder** + **MMDiT transformer** + **Autoencoder VAE**
* Uses `ZipDataset` for dynamic dataset loading and prompt embedding
* Live training tracked via a `progress.json` file

---

## 📜 License

[MIT](LICENSE) © Black Forest Labs

---

## 🙋‍♂️ Acknowledgments

* [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* [Gradio](https://gradio.app/)
* [LoRA (PEFT)](https://github.com/huggingface/peft)
* [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
* [Accelerate](https://github.com/huggingface/accelerate)
