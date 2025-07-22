import argparse
import json
import os
import sys
import torch
import copy
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers import AutoencoderKL, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model_state_dict
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
import bitsandbytes as bnb

import os
from pathlib import Path

from zipdataset import ZipDataset


FOUNDATIONAL_MODEL = os.environ.get("FOUNDATIONAL_MODEL")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--progress_file", type=str, help="File to write progress updates")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Extract config values
    adapter_name = config["adapter_name"]
    trigger_word = config["trigger_word"] 
    lora_rank = int(config["lora_rank"])
    dataset_zip_path = config["dataset_zip_path"]
    use_8bit_adam = config["use_8bit_adam"]
    gradient_checkpointing = config["gradient_checkpointing"]
    cache_latents = config["cache_latents"]
    use_quantization = config["use_quantization"]
    max_train_steps = int(config["max_train_steps"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
    train_batch_size = int(config["train_batch_size"])
    guidance_scale = float(config["guidance_scale"])
    learning_rate = float(config["learning_rate"])
    
    # Training hyperparameters
    mixed_precision = "fp16"
    width, height = 512, 768
    weight_decay = 1e-04
    eps = 1e-08
    warmup_steps = 0
    max_sequence_length = 100
    output_dir = "lora_output"
    
    try:
        # Setup accelerator for multi-GPU FIRST
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            project_config=ProjectConfiguration(
                project_dir=output_dir, 
                logging_dir=Path(output_dir, "logs")
            ),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
        )
        
        # Define write_progress function
        def write_progress(progress, message):
            """Write progress to file for Gradio to read - ONLY from main process"""
            if accelerator.is_main_process:
                if args.progress_file:
                    existing_data = {"progress": 0.0, "messages": []}
                    try:
                        if os.path.exists(args.progress_file):
                            with open(args.progress_file, "r") as f:
                                existing_data = json.load(f)
                    except:
                        pass
                    
                    new_progress = max(progress, existing_data.get("progress", 0.0))
                    messages = existing_data.get("messages", [])
                    messages.append(f"[{new_progress:.1%}] {message}")
                    
                    # Keep ALL messages - removed truncation
                    progress_data = {
                        "progress": new_progress, 
                        "messages": messages,
                        "latest_message": message
                    }
                    
                    temp_file = args.progress_file + ".tmp"
                    with open(temp_file, "w") as f:
                        json.dump(progress_data, f)
                    os.rename(temp_file, args.progress_file)
            
            # Only main process prints progress to avoid spam
            if accelerator.is_main_process:
                print(f"[GPU {accelerator.process_index}/{accelerator.num_processes}] [{progress:.2%}] {message}", flush=True)
        
        write_progress(0.05, f"Initializing distributed training on {accelerator.num_processes} GPU(s)...")
        
        # COORDINATE MODEL LOADING - Only main process downloads, others wait
        with accelerator.main_process_first():
            write_progress(0.1, "Loading scheduler from foundational model...")
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                FOUNDATIONAL_MODEL, 
                subfolder="scheduler"
            )
            noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        # Wait for all processes to reach this point
        accelerator.wait_for_everyone()
        write_progress(0.12, "Scheduler loaded successfully on all processes")
        
        with accelerator.main_process_first():
            write_progress(0.15, "Loading VAE from foundational model...")
            vae = AutoencoderKL.from_pretrained(
                FOUNDATIONAL_MODEL, 
                subfolder="vae"
            )
        
        accelerator.wait_for_everyone()
        write_progress(0.18, "VAE loaded successfully on all processes")
        
        # Setup quantization
        nf4_config = None
        if use_quantization:
            if accelerator.is_main_process:
                write_progress(0.2, "Setting up 4-bit quantization with NF4...")
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.float16
            )
            write_progress(0.22, f"Quantization config created - NF4 4-bit with FP16 compute dtype")
        else:
            write_progress(0.22, "Quantization disabled - using full precision")
        

        accelerator.wait_for_everyone()
        write_progress(0.29, "Preparing quantized model for k-bit training...")
        # Load transformer with coordination
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                write_progress(0.25, "Loading Flux transformer model...")
            
            transformer = FluxTransformer2DModel.from_pretrained(
                FOUNDATIONAL_MODEL, 
                subfolder="transformer",
                quantization_config=nf4_config, 
                torch_dtype=torch.float16
            )
                        
            if use_quantization:                
                # Check if the model has get_input_embeddings method
                if not hasattr(transformer, 'get_input_embeddings'):
                    if accelerator.is_main_process:
                        write_progress(0.295, "Model missing get_input_embeddings - adding custom implementation...")
                    
                    # Add the missing method to make prepare_model_for_kbit_training work
                    def get_input_embeddings(self):
                        # For Flux models, return None or the first embedding layer we can find
                        for module in self.modules():
                            if hasattr(module, 'weight') and 'embed' in type(module).__name__.lower():
                                return module
                        return None
                    
                    def set_input_embeddings(self, value):
                        # Dummy implementation - not used in our case
                        pass
                    
                    # Monkey patch the methods onto the model
                    import types
                    transformer.get_input_embeddings = types.MethodType(get_input_embeddings, transformer)
                    transformer.set_input_embeddings = types.MethodType(set_input_embeddings, transformer)
                    
                    write_progress(0.297, "Custom get_input_embeddings method added successfully")
                
                # Now this should work
                transformer = prepare_model_for_kbit_training(
                    transformer, 
                    use_gradient_checkpointing=False
                )
                
                write_progress(0.3, "Model prepared for k-bit training successfully")
        
        accelerator.wait_for_everyone()
        write_progress(0.3, "Transformer loaded successfully on all processes")

        # Move models to device
        write_progress(0.31, f"Moving VAE to device: {accelerator.device}")
        vae.requires_grad_(False)
        vae.to(accelerator.device, dtype=torch.float16)
        write_progress(0.32, "VAE moved to device and gradients disabled")
        
        # Setup LoRA
        write_progress(0.33, "Setting up LoRA configuration...")
        transformer.requires_grad_(False)
        write_progress(0.34, "Transformer gradients disabled")
        
        if gradient_checkpointing:
            write_progress(0.35, "Enabling gradient checkpointing for memory efficiency...")
            transformer.enable_gradient_checkpointing()
            write_progress(0.36, "Gradient checkpointing enabled")
        else:
            write_progress(0.36, "Gradient checkpointing disabled")
        
        write_progress(0.37, f"Creating LoRA configuration with rank {lora_rank}")
        transformer_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        write_progress(0.38, "Adding LoRA adapter to transformer...")
        transformer.add_adapter(transformer_lora_config, adapter_name=adapter_name)
        write_progress(0.39, f"LoRA adapter added - targeting modules: {transformer_lora_config.target_modules}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        write_progress(0.4, f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Setup optimizer
        if use_8bit_adam:
            write_progress(0.41, "Setting up 8-bit AdamW optimizer...")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            write_progress(0.41, "Setting up standard AdamW optimizer...")
            optimizer_class = torch.optim.AdamW
        
        trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        write_progress(0.42, f"Found {len(trainable_parameters)} trainable parameter groups")
        
        optimizer = optimizer_class(
            [{"params": trainable_parameters, "lr": learning_rate}],
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=eps
        )
        write_progress(0.43, f"Optimizer created - LR: {learning_rate}, Weight Decay: {weight_decay}, Eps: {eps}")
        
        # Setup dataset - each process will handle its own shard
        write_progress(0.44, f"Loading dataset from: {dataset_zip_path}")
        dataset = ZipDataset(
            dataset_zip_path, 
            width=width, 
            height=height, 
            max_sequence_length=max_sequence_length, 
            model_id=FOUNDATIONAL_MODEL,
            trigger_word=trigger_word,
            accelerator_device=accelerator.device
        )
        write_progress(0.45, f"Dataset loaded - {len(dataset)} samples, resolution: {width}x{height}")
        
        write_progress(0.46, "Creating data loader...")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            collate_fn=ZipDataset.collate_fn,
            num_workers=0,  # Keep 0 for CUDA safety
            pin_memory=True,
            drop_last=True  # Important for multi-GPU training
        )
        write_progress(0.47, f"Data loader created - batch size: {train_batch_size}, shuffle: True, drop_last: True")
        
        # Setup scheduler
        write_progress(0.48, "Calculating training schedule...")
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        max_train_steps = int(max_train_steps)
        write_progress(0.49, f"Using specified max train steps: {max_train_steps}")
        
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        write_progress(0.5, f"Training schedule: {num_train_epochs} epochs, {num_update_steps_per_epoch} steps per epoch")
        
        write_progress(0.51, "Creating learning rate scheduler...")
        lr_scheduler = get_scheduler(
            "constant", 
            optimizer=optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=max_train_steps
        )
        write_progress(0.52, f"LR scheduler created - type: constant, warmup steps: {warmup_steps}")
        
        # Prepare for training - CRITICAL for multi-GPU
        write_progress(0.53, "Preparing models and data for distributed training...")
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
        write_progress(0.54, "Models and data prepared for distributed training")
        
        # Disable latent caching for multi-GPU to avoid complications
        if accelerator.num_processes > 1:
            cache_latents = False
            write_progress(0.55, f"Disabled latent caching for multi-GPU training ({accelerator.num_processes} processes)")
        else:
            write_progress(0.55, f"Single GPU training - latent caching {'enabled' if cache_latents else 'disabled'}")
        
        # Get VAE config
        vae_config = vae.config
        write_progress(0.56, f"VAE config retrieved - shift factor: {vae_config.shift_factor}, scaling factor: {vae_config.scaling_factor}")
        
        # Free VAE memory if not caching
        if not cache_latents:
            write_progress(0.57, "Freeing VAE memory (will reload for on-the-fly encoding)...")
            del vae
            free_memory()
            write_progress(0.58, "VAE memory freed")
        else:
            write_progress(0.58, "Keeping VAE in memory for latent caching")
        
        # Training setup
        write_progress(0.59, "Setting up training utilities...")
        
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            return model._orig_mod if is_compiled_module(model) else model
        
        if mixed_precision == "fp16":
            write_progress(0.6, "Casting training parameters to FP32 for FP16 mixed precision...")
            cast_training_params([transformer], dtype=torch.float32)
            write_progress(0.61, "Training parameters cast to FP32")
        else:
            write_progress(0.61, f"Mixed precision: {mixed_precision} - no parameter casting needed")
        
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(accelerator.device)]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        global_step = 0
        write_progress(0.62, "Training utilities setup complete")
        
        # Training loop
        write_progress(0.63, f"Starting training loop on {accelerator.num_processes} GPU(s)...")
        write_progress(0.64, f"Training configuration: {num_train_epochs} epochs, {max_train_steps} max steps, gradient accumulation: {gradient_accumulation_steps}")
        
        # Create VAE for on-the-fly encoding if needed
        if not cache_latents:
            write_progress(0.65, "Loading VAE for on-the-fly encoding...")
            with accelerator.main_process_first():
                vae = AutoencoderKL.from_pretrained(FOUNDATIONAL_MODEL, subfolder="vae")
            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=torch.float16)
            write_progress(0.66, "VAE loaded for on-the-fly encoding")
        
        for epoch in range(num_train_epochs):
            epoch_start_progress = 0.67 + (epoch / num_train_epochs) * 0.28  # 67% to 95%
            write_progress(epoch_start_progress, f"Starting epoch {epoch + 1}/{num_train_epochs}")
            
            epoch_loss_sum = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(transformer):
                    # Encode images on-the-fly
                    with torch.no_grad():
                        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                    
                    model_input = (model_input - vae_config.shift_factor) * vae_config.scaling_factor
                    model_input = model_input.to(dtype=torch.float16)
                    
                    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2,
                        accelerator.device, torch.float16
                    )
                    
                    noise = torch.randn_like(model_input, device=accelerator.device, dtype=torch.float16)
                    bsz = model_input.shape[0]
                    u = compute_density_for_timestep_sampling("none", bsz, 0.0, 1.0, 1.29)
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                    
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                    
                    packed_noisy_model_input = FluxPipeline._pack_latents(
                        noisy_model_input, model_input.shape[0], model_input.shape[1],
                        model_input.shape[2], model_input.shape[3]
                    )
                    
                    guidance = None
                    if unwrap_model(transformer).config.guidance_embeds:
                        guidance = torch.tensor([guidance_scale], device=accelerator.device, dtype=torch.float16).expand(bsz)
                    
                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=batch["pooled_prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                        encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                        txt_ids=batch["text_ids"].to(accelerator.device, dtype=torch.float16),
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                    
                    vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1)
                    model_pred = FluxPipeline._unpack_latents(
                        model_pred, model_input.shape[2] * vae_scale_factor,
                        model_input.shape[3] * vae_scale_factor, vae_scale_factor
                    )
                    
                    weighting = compute_loss_weighting_for_sd3("none", sigmas)
                    target = noise - model_input
                    loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
                    epoch_loss_sum += loss.detach().item()
                    epoch_steps += 1
                    
                    # Log every 10 steps or at important milestones
                    if accelerator.is_main_process and (global_step % 10 == 0 or global_step == 1 or global_step % 50 == 0):
                        step_progress = epoch_start_progress + ((step + 1) / len(train_dataloader)) * (0.28 / num_train_epochs)
                        avg_loss = epoch_loss_sum / epoch_steps
                        current_lr = lr_scheduler.get_last_lr()[0]
                        
                        message = f"Epoch {epoch + 1}/{num_train_epochs}, Step {global_step}/{max_train_steps} - Loss: {loss.detach().item():.6f}, Avg Loss: {avg_loss:.6f}, LR: {current_lr:.8f}"
                        write_progress(step_progress, message)
                        
                        # Memory usage info every 50 steps
                        if global_step % 50 == 0:
                            if torch.cuda.is_available():
                                memory_allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                                memory_reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                                write_progress(step_progress + 0.001, f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
                if global_step >= max_train_steps:
                    write_progress(0.93, f"Reached maximum training steps ({max_train_steps}), stopping training")
                    break
            
            if global_step >= max_train_steps:
                break
                
            # End of epoch summary
            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_steps
                epoch_end_progress = 0.67 + ((epoch + 1) / num_train_epochs) * 0.28
                write_progress(epoch_end_progress, f"Completed epoch {epoch + 1}/{num_train_epochs} - Average loss: {avg_epoch_loss:.6f}, Steps: {epoch_steps}")
        
        # Save model
        write_progress(0.95, "Training completed, saving LoRA weights...")
        accelerator.wait_for_everyone()
        write_progress(0.96, "All processes synchronized, proceeding with model saving...")
        
        if accelerator.is_main_process:
            write_progress(0.97, "Extracting LoRA state dict from trained model...")
            transformer_lora_layers = get_peft_model_state_dict(unwrap_model(transformer), adapter_name=adapter_name)
            write_progress(0.98, f"LoRA state dict extracted, saving to {output_dir}...")
            FluxPipeline.save_lora_weights(
                output_dir, 
                transformer_lora_layers=transformer_lora_layers, 
                text_encoder_lora_layers=None
            )
            write_progress(0.99, f"LoRA weights saved successfully to {os.path.abspath(output_dir)}")
        
        accelerator.wait_for_everyone()
        
        # Final statistics
        if accelerator.is_main_process:
            total_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in transformer.parameters())
            final_lr = lr_scheduler.get_last_lr()[0]
            
            write_progress(1.0, f"Training complete! Final statistics:")
            write_progress(1.0, f"  - Total steps: {global_step}")
            write_progress(1.0, f"  - Epochs completed: {epoch + 1}")
            write_progress(1.0, f"  - Final learning rate: {final_lr:.8f}")
            write_progress(1.0, f"  - Trainable parameters: {total_trainable_params:,} / {total_params:,} ({100 * total_trainable_params / total_params:.2f}%)")
            write_progress(1.0, f"  - Model saved to: {os.path.abspath(output_dir)}")
            write_progress(1.0, f"  - Training completed successfully on {accelerator.num_processes} GPU(s)")
        
        accelerator.wait_for_everyone()
        accelerator.end_training()
        sys.exit(0)
    except Exception as e:
        import traceback
        error_msg = f"Training failed with error: {str(e)}"
        full_traceback = traceback.format_exc()
        
        if 'accelerator' in locals():
            if accelerator.is_main_process:
                write_progress(0.0, error_msg)
                write_progress(0.0, f"Full traceback: {full_traceback}")
                print(f"ERROR: {error_msg}", flush=True)
                print(f"TRACEBACK:\n{full_traceback}", flush=True)
                
                if args.progress_file:
                    try:
                        error_data = {
                            "progress": 0.0, 
                            "messages": [error_msg, f"Traceback: {full_traceback}"], 
                            "latest_message": error_msg,
                            "error": True
                        }
                        with open(args.progress_file, "w") as f:
                            json.dump(error_data, f)
                    except Exception as save_error:
                        print(f"Failed to save error to progress file: {save_error}", flush=True)
        else:
            print(f"ERROR (before accelerator init): {error_msg}", flush=True)
            print(f"TRACEBACK:\n{full_traceback}", flush=True)
        
        sys.exit(1)

if __name__ == "__main__":
    main()