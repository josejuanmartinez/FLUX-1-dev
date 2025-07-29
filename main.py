import gradio as gr
import torch
import os, json

from transformers import T5EncoderModel

from diffusers import (
    BitsAndBytesConfig,
    FluxPipeline, FluxTransformer2DModel,
)

import tempfile

import os
import tempfile

import gc

import subprocess
import json
import tempfile
import time
import os

# Constants
FOUNDATIONAL_MODEL = os.environ.get("FOUNDATIONAL_MODEL")
GPU_INFERENCE=False
train_use_quantization = None

# Global
busy = False

# Model configuration
mixed_precision="fp16"
width, height = 256, 512

# Optimizer
weight_decay=1e-04
eps=1e-08
warmup_steps=0

# Inference
max_sequence_length=100

# Output directory for training
output_dir = "lora_output"

# Generator for reproducibility
generator = torch.Generator("cuda").manual_seed(0)
generator_cpu = torch.Generator("cpu").manual_seed(0)

# FOUNDATIONAL (FLUX-1 Dev)
# Load model ONCE at startup
# FLUX-1 Dev Foundational Model
# Architecture
# - The model consists of three main components:
# 1) Text Encoders (CLIP and T5)
# 2) Transformer (Main Model - MMDiT)
# 3) Variational Auto-Encoder (VAE)
# In QLoRA approach, we focus exclusively on fine-tuning the transformer component (MMDiT). 
# The text encoders and VAE remain frozen throughout training.
foundational_pipe = FluxPipeline.from_pretrained(
    FOUNDATIONAL_MODEL,
    torch_dtype=torch.float32,  # Use float32 for CPU
)
# Ensure everything is on CPU
foundational_pipe = foundational_pipe.to("cpu")

# GRADIO HOOKS
def infer_foundational_gpu(prompt: str, inference_steps: int):
    """
    Run inference on the foundational model using GPU.
    This function checks if the model is busy, and if not, it performs inference on the GPU.
    """
    if busy: 
        return None
    
    print(f"Starting foundational inference on GPU with {inference_steps} steps", flush=True)
    
    torch.cuda.empty_cache()
    
    try:       
        image = foundational_pipe(
            prompt=prompt,
            prompt_2=None,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=int(inference_steps),
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]

        foundational_pipe.to("cuda")
        foundational_pipe.enable_model_cpu_offload()
        
        torch.cuda.empty_cache()
        
        print(f"Foundational inference completed on GPU", flush=True)
        return image
        
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"Error in foundational inference: {e}", flush=True)
        return None

def infer_foundational(prompt: str, inference_steps: int):
    """ Run inference on the foundational model using CPU.
    This function checks if the model is busy, and if not, it performs inference on the CPU.
    """
    if busy: 
        return None
    
    print(f"Starting foundational inference on CPU with {inference_steps} steps", flush=True)
    
    torch.cuda.empty_cache()
    
    try:       
        image = foundational_pipe(
            prompt=prompt,
            prompt_2=None,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=int(inference_steps),
            max_sequence_length=max_sequence_length,
            generator=generator_cpu
        ).images[0]
        
        torch.cuda.empty_cache()
        
        print(f"Foundational inference completed on CPU", flush=True)
        return image
        
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"Error in foundational inference: {e}", flush=True)
        return None

def infer_adapted(prompt, inference_steps: int):
    """ Run inference on an adapted model (LoRA) using the appropriate device.
    This function checks if the model is busy, and if not, it performs inference on either CPU or GPU.
    """
    if busy: return
    
    ckpt_id = FOUNDATIONAL_MODEL
    lora_path = output_dir
    if not os.path.exists(lora_path):
        print(f"Expected model not found at: {lora_path}", flush=True)
        return
    
    print(f"Starting adapted inference with {inference_steps} steps", flush=True)
    
    # Remove quantization for CPU inference - it's not well supported
    print("Loading the transformer for CPU", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        ckpt_id, 
        subfolder="transformer",
        torch_dtype=torch.float32  # Use float32 for better CPU performance
    )

    print("Loading the Text Encoder", flush=True)
    text_encoder = T5EncoderModel.from_pretrained(
        ckpt_id, 
        subfolder="text_encoder_2", 
        torch_dtype=torch.float32  # Use float32 for CPU
    )

    print("Loading the Pipeline", flush=True)
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )

    try:        
        pipeline.load_lora_weights(lora_path)   
        print(f"[INFO] Successfully loaded LoRA weights from: {os.path.abspath(lora_path)}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load LoRA weights from: {lora_path}", flush=True)
        return
    
    # Clean up individual components
    del text_encoder
    del transformer
    gc.collect()

    # Move pipeline to CPU
    pipeline.to("cpu")
    
    # Optional: Enable CPU optimizations
    # pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")
    
    image = pipeline(
        prompt,
        num_inference_steps=int(inference_steps),
        guidance_scale=3.5,
        height=height,
        width=width,
        generator=generator_cpu
    ).images[0]

    # Remove CUDA memory check for CPU inference
    print("Inference completed on CPU")

    return image

def infer_adapted_gpu(prompt, inference_steps: int):
    """ Run inference on an adapted model (LoRA) using GPU.
    This function checks if the model is busy, and if not, it performs inference on the GPU.
    """
    if busy: return
    global train_use_quantization
    torch.cuda.empty_cache()

    ckpt_id = FOUNDATIONAL_MODEL
    lora_path = output_dir
    if not os.path.exists(lora_path):
        print (f"Expected model not found at: {lora_path}", flush=True)
        return
    
    print(f"Starting adapted GPU inference with {inference_steps} steps", flush=True)
    
    nf4_config = None
    if train_use_quantization:
        print("Setting up quantization", flush=True)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16
        )
    
    print("Loading the transformer", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        ckpt_id, 
        subfolder="transformer",
        quantization_config=nf4_config, 
        torch_dtype=torch.float16
    )

    print("Loading the Text Encoder", flush=True)
    # text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder_2", quantization_config=nf4_config, torch_dtype=torch.float16,)
    text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.float16)

    print("Loading the Pipeline", flush=True)
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=torch.float16,
    )

    try:        
        pipeline.load_lora_weights(lora_path)   
        print(f"[INFO] Successfully loaded LoRA weights from: {os.path.abspath(lora_path)}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load LoRA weights from: {lora_path}", flush=True)
        return
    
    del text_encoder
    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()

    image = pipeline(
        prompt,
        num_inference_steps=int(inference_steps),
        guidance_scale=3.5,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

    return image

def train_adapter(
    adapter_name: str,
    trigger_word: str,
    lora_rank: int,
    dataset_zip,
    use_8bit_adam: bool,
    gradient_checkpointing: bool,
    cache_latents: bool,
    use_quantization: bool,
    max_train_steps: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    train_batch_size: int,
    guidance_scale: float
):
    """ Train a LoRA adapter using the provided parameters.
    This function checks if the training is already in progress, and if not, it starts the training process.
    It uses a temporary configuration file and monitors the training progress.
    """
    global busy, train_use_quantization
    if busy: return
    print("Starting adapter training...", flush=True)
    if not hasattr(dataset_zip, "name") or not os.path.isfile(dataset_zip.name):
        yield 0.0, "Error: Invalid file upload"
        return
    
    train_use_quantization = use_quantization
    
    busy = True
    # Create temporary config file
    config = {
        "adapter_name": adapter_name,
        "trigger_word": trigger_word,
        "lora_rank": lora_rank,
        "dataset_zip_path": dataset_zip.name,
        "use_8bit_adam": use_8bit_adam,
        "gradient_checkpointing": gradient_checkpointing,
        "cache_latents": cache_latents,
        "use_quantization": use_quantization,
        "max_train_steps": max_train_steps,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_batch_size": train_batch_size,
        "guidance_scale": guidance_scale
    }
    
    # Create temporary files
    process = None
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    progress_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)        
    
    try:
        # Write config
        json.dump(config, config_file)
        config_file.close()
        progress_file.close()
        
        # Launch training with accelerate
        cmd = [
            "accelerate", "launch",
            "--config_file", "accelerate_config.yaml",  # Use your config file
            "train.py",
            "--config", config_file.name,
            "--progress_file", progress_file.name
        ]
        
        # Start subprocess
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor progress
        last_progress = 0.0
        all_messages = []
        console_output = []
        
        while process.poll() is None:
            # Read progress from file
            try:
                if os.path.exists(progress_file.name):
                    with open(progress_file.name, 'r') as f:
                        progress_data = json.load(f)
                        
                        # Get progress (only moves forward)
                        new_progress = progress_data.get("progress", last_progress)
                        if new_progress > last_progress:
                            last_progress = new_progress
                        
                        # Get accumulated messages
                        file_messages = progress_data.get("messages", [])
                        
                        # Merge with any new messages we haven't seen
                        if len(file_messages) > len(all_messages):
                            all_messages = file_messages
                            
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            
            # Read any stdout output (from all processes)
            try:
                line = process.stdout.readline()
                if line and line.strip():
                    console_output.append(line.strip())
                    # Keep only last 30 console lines
                    if len(console_output) > 30:
                        console_output = console_output[-30:]
            except:
                pass
            
            # Combine progress messages and console output
            display_text = "\n".join(all_messages)
            if console_output:
                display_text += "\n\n--- Console Output ---\n" + "\n".join(console_output[-10:])  # Show last 10 console lines
            
            yield last_progress, display_text
        
        # Process finished - get final result
        return_code = process.returncode
        
        # Read final progress
        try:
            if os.path.exists(progress_file.name):
                with open(progress_file.name, 'r') as f:
                    progress_data = json.load(f)
                    last_progress = progress_data.get("progress", last_progress)
                    all_messages = progress_data.get("messages", all_messages)
        except:
            pass
        
        # Read any remaining output
        remaining_output = process.stdout.read()
        if remaining_output:
            remaining_lines = remaining_output.strip().split('\n')
            console_output.extend(remaining_lines)
        
        # Final display
        final_text = "\n".join(all_messages)
        if console_output:
            final_text += "\n\n--- Console Output ---\n" + "\n".join(console_output[-10:])
        
        if return_code == 0:
            final_text += "\n\n✅ Training completed successfully!"
            yield 1.0, final_text
        else:
            final_text += f"\n\n❌ Training failed with return code {return_code}"
            yield 0.0, final_text        
    finally:
        # Cleanup process
        if process:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    time.sleep(2)
                    if process.poll() is None:  # Still running, force kill
                        process.kill()
                process.wait()  # Ensure process is cleaned up
            except:
                pass
            
            try:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            except:
                pass
        
        # Cleanup temp files
        for temp_file in [config_file, progress_file]:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        # Reset busy flag
        busy = False

def run():
    """ Build and launch the Gradio UI for FLUX-1 Developer Hub.
    This function sets up the UI components, defines the interactions, and starts the Gradio server.
    """
    print("Building UI...", flush=True)
    with gr.Blocks(theme=gr.themes.Monochrome(), css="styles.css") as demo:
        gr.Markdown("# 🚀 FLUX-1 Developer Hub")
        gr.Markdown("A unified UI to explore **FLUX-1 Dev**, fine-tune adapters, and run inference.")

        with gr.Tabs():
            with gr.TabItem("🔍 Foundational Inference"):
                with gr.Box():
                    gr.Markdown("### Base Model Inference")
                    with gr.Row():
                        with gr.Column(scale=1):
                            foundational_prompt = gr.Textbox(
                                label="📝 Prompt",
                                placeholder="Describe what you want to generate..."
                            )
                            foundational_inference_steps = gr.Number(
                                label="🔄 Inference Steps",
                                value=15,
                                minimum=1,
                                precision=0,
                                maximum=100,
                                step=1
                            )
                            foundational_run_btn = gr.Button("✨ Run Inference")
                        with gr.Column(scale=1):
                            foundational_img_out = gr.Image(label="🎨 Output", type="pil", show_label=True)
                    foundational_run_btn.click(
                        infer_foundational if not GPU_INFERENCE else infer_foundational_gpu,
                        inputs=[foundational_prompt, foundational_inference_steps], 
                        outputs=foundational_img_out
                    )

            with gr.TabItem("🧪 Train Adapter"):
                with gr.Box():
                    gr.Markdown("### Fine-tune a LoRA Adapter on FLUX-1")
                    with gr.Row():
                        with gr.Column():
                            adapter_name = gr.Textbox(
                                label="📛 Adapter Name", 
                                value="retro",
                                info="Name for your LoRA adapter (like a filename). Choose something descriptive."
                            )

                            trigger_word = gr.Textbox(
                                label="🔑 Trigger Word", 
                                value="<retro>",
                                info="Special word to activate your style. Use <brackets> to avoid conflicts."
                            )

                            lora_rank = gr.Number(
                                label="📐 LoRA Rank", 
                                value=4,
                                info="Adapter complexity. Lower (1-8) = simpler/faster. Higher (16-64) = more detailed but slower. 4-16 is usually good."
                            )

                            max_train_steps = gr.Number(
                                label="🏁 Training Steps", 
                                value=100,
                                info="How long to train. More steps = better learning but risk overfitting. Start with 500-1500 for small datasets."
                            )
                            dataset_zip = gr.File(label="🗂️ Dataset (ZIP)", file_types=[".zip"])

                            gr.Markdown("#### 📊 Training Hyperparameters")
                            learning_rate = gr.Number(
                                label="📚 Learning Rate", 
                                value=1e-4, 
                                minimum=1e-6, 
                                maximum=1e-2, 
                                step=1e-6,
                                info="How fast the model learns. Lower = slower but more stable. Higher = faster but may be unstable. 1e-4 is good for LoRA."
                            )

                            train_batch_size = gr.Number(
                                label="📦 Train Batch Size", 
                                value=1,
                                info="Samples per GPU step. Higher = more GPU memory used."
                            )

                            gradient_accumulation_steps = gr.Number(
                                label="📈 Gradient Accumulation Steps", 
                                value=1,
                                info="Steps before weight update. Effective batch size = batch_size × accumulation_steps"
                            )

                            guidance_scale = gr.Number(
                                label="🎯 Guidance Scale", 
                                value=3.5, 
                                minimum=1.0, 
                                maximum=20.0, 
                                step=0.1,
                                info="How strictly to follow prompts. Lower = more creative. Higher = more prompt-faithful. 3.5-5.0 is balanced."
                            )

                            gr.Markdown("#### ⚙️ Performance Options")
                            use_8bit_adam = gr.Checkbox(
                                label="🧠 Use 8-bit Adam", 
                                value=True,
                                info="Memory-efficient optimizer. ✅ Reduces GPU memory by ~50% with minimal quality loss. Recommended for most users."
                            )
                            gradient_checkpointing = gr.Checkbox(
                                label="📉 Gradient Checkpointing", 
                                value=True,
                                info="Trades speed for memory. ✅ Prevents out-of-memory errors. ❌ Training ~10-20% slower. Keep checked if low on GPU memory."
                            )
                            cache_latents = gr.Checkbox(
                                label="💾 Cache Latents", 
                                value=True,
                                info="Pre-computes image encodings. ✅ Significantly faster training. ❌ Uses more disk space. Recommended unless storage is very limited."
                            )
                            use_quantization = gr.Checkbox(
                                label="⚡ Use Quantization", 
                                value=True,
                                info="Loads model in 4-bit precision. ✅ Reduces GPU memory by ~75%. Best for consumer GPUs (≤16GB VRAM). ❌ Uncheck for high-end GPUs if you want maximum quality."
                            )

                            train_btn = gr.Button("🎯 Train Adapter")
                            # NEW: Progress bar and log
                            train_progress_bar = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.01,
                                value=0.0, label="📈 Progress", interactive=False
                            )
                            train_output = gr.Textbox(label="📢 Output Log", lines=8, interactive=True)

                            train_btn.click(
                                train_adapter,
                                inputs=[
                                    adapter_name,
                                    trigger_word,
                                    lora_rank,
                                    dataset_zip,
                                    use_8bit_adam,
                                    gradient_checkpointing,
                                    cache_latents,
                                    use_quantization,
                                    max_train_steps,
                                    learning_rate,
                                    gradient_accumulation_steps,
                                    train_batch_size,
                                    guidance_scale
                                ],
                                outputs=[train_progress_bar, train_output],
                                show_progress=True
                            )

            with gr.TabItem("🧠 Adapted Inference"):
                with gr.Box():
                    gr.Markdown("### Inference with a Fine-tuned Adapter")
                    with gr.Row():
                        with gr.Column(scale=1):
                            adapted_prompt = gr.Textbox(
                                label="📝 Prompt",
                                placeholder="Use your trigger word to invoke the adapter..."
                            )
                            adapted_inference_steps = gr.Number(
                                label="🔄 Inference Steps",
                                value=15,
                                minimum=1,
                                precision=0,
                                maximum=100,
                                step=1
                            )
                            adapted_run_btn = gr.Button("🚀 Run Inference")
                        with gr.Column(scale=1):
                            adapted_img_out = gr.Image(label="🎨 Output", type="pil", show_label=True)
                    adapted_run_btn.click(
                        infer_adapted if not GPU_INFERENCE else infer_adapted_gpu, 
                        inputs=[adapted_prompt, adapted_inference_steps], 
                        outputs=adapted_img_out
                    )
        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

if __name__ == "__main__":
    print("Running Gradio app...")
    run()
