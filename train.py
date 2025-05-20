import gc
import os
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, DataCollatorForLanguageModeling
from transformers.trainer_utils import speed_metrics
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM 

from ensemble import ModelEnsemble
import config

# debug
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Makes CUDA errors synchronous and easier to debug

teacher_model = None
ensemble_model = None

def format_time_elapsed(seconds):
    """Convert seconds to a readable format with minutes and seconds."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"

def get_round_path(output_path, round_num):
    """Return path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")

def evaluate_model(model, eval_dataset, device, collate_fn):
    """Evaluates a model and returns the language modeling loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    # Create a DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Temporary bandaid
            if "labels" not in batch:
                labels = input_ids.clone()
            else:
                labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return {"eval_loss": avg_loss}

def cleanup_memory():
    """Helper function to clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.round_num = kwargs.pop("round_num")
        self.steps_per_round = kwargs.pop("steps_per_round")
        super().__init__(*args, **kwargs)

    def compute_kl_loss(self, student_logits, ensemble_logits, teacher_logits, mask, temperature=1.0):
        """Computes KL divergence loss between teacher and student model logits."""
        teacher_logits = teacher_logits.to(student_logits.device)
        if ensemble_logits is not None:
            num_models = len(ensemble_model.models)
            ensemble_logits = ensemble_logits.to(student_logits.device)
            student_logits = student_logits/(num_models + 1) + ensemble_logits*(num_models/(num_models + 1))
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        forward_kl = F.kl_div(student_probs, teacher_probs, log_target=True, reduction="none").sum(-1)
        return forward_kl[mask].mean()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss function for knowledge distillation."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        student_device = next(model.parameters()).device
        teacher_device = next(teacher_model.parameters()).device
        
        # Process teacher and ensemble logits
        with torch.no_grad():
            teacher_inputs = {
                "input_ids": input_ids.to(teacher_device),
                "attention_mask": attention_mask.to(teacher_device)
            }
            teacher_logits = teacher_model(**teacher_inputs).logits
            
            if ensemble_model is not None:
                ensemble_device = next(ensemble_model.parameters()).device
                ensemble_inputs = {
                    "input_ids": input_ids.to(ensemble_device),
                    "attention_mask": attention_mask.to(ensemble_device)
                }
                ensemble_logits = ensemble_model(**ensemble_inputs).logits
                ensemble_logits = ensemble_logits.to(student_device)
            else:
                ensemble_logits = None
        
        # Process student logits
        student_inputs = {
            "input_ids": input_ids.to(student_device),
            "attention_mask": attention_mask.to(student_device)
        }
        student_logits = model(**student_inputs).logits
        
        # Get teacher logits to student device for loss computation
        teacher_logits = teacher_logits.to(student_device)
        
        # Calculate loss
        loss = self.compute_kl_loss(
            student_logits, 
            ensemble_logits, 
            teacher_logits, 
            labels.to(student_device) != -100
        )
        
        return (loss, student_logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        student_device = next(model.parameters()).device
        
        with torch.no_grad():
            student_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            if ensemble_model is not None:
                num_models = len(ensemble_model.models)
                ensemble_logits = ensemble_model(input_ids=inputs["input_ids"].to(student_device), attention_mask=inputs["attention_mask"].to(student_device)).logits.to(student_device)
                student_logits = (student_logits/(num_models + 1) + ensemble_logits*(num_models/(num_models+1))).detach()

        # Check if model has module attribute (DP/DDP wrapped) or not
        if hasattr(model, "module"):
            actual_model = model.module
        else:
            actual_model = model
        
        # Use model's loss function if it exists, otherwise use standard CE loss
        if hasattr(actual_model, "loss_function"):
            config = actual_model.config
            loss = actual_model.loss_function(logits=student_logits, labels=inputs["labels"], vocab_size=config.vocab_size).detach()
        else:
            # Standard language modeling loss
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            # Only calculate loss on non-padded tokens
            active_loss = shift_labels != -100
            if active_loss.any():
                active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss.view(-1)]
                active_labels = shift_labels.view(-1)[active_loss.view(-1)]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0).to(student_device)

        return (loss, None, None) if prediction_loss_only else (loss, student_logits, inputs["labels"])
    
    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step + (self.round_num*self.steps_per_round)}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

# Define a callback to save evaluation metrics to wandb
class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)

def main():
    global teacher_model, ensemble_model
    
    # Record start time
    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting training at: {overall_start_datetime}")

    # Get run directory
    output_path = config.get_run_directory()
    print(f"Models stored in: {output_path}")

    # Setup wandb
    run_name = f"{os.path.basename(output_path)}"
    wandb.init(project="slm_ensembles", name=run_name)
    wandb.config.update({
        "student_model": config.student_model_name,
        "teacher_model": config.teacher_model_name,
        "total_rounds": config.total_rounds,
        "steps_per_round": config.steps_per_round,
        "training_start_time": overall_start_datetime
    })

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    teacher_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
    teacher_model.requires_grad_(False)

    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Evaluate the teacher model at the beginning
    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], teacher_model.device, collator)
    wandb.log({
        "teacher/eval_loss": teacher_eval_results["eval_loss"],
        "round": -1  # To indicate a pre-training baseline
    })
    print(f"Teacher model evaluation: {teacher_eval_results}")
    
    # Crash recovery code - check for existing ensembles and setup ensemble
    existing_rounds = []
    for i in range(config.total_rounds):
        round_path = get_round_path(output_path, i)
        if os.path.exists(round_path):
            existing_rounds.append(i)
            
    # If we have existing rounds, load the ensemble
    ensemble_model_names = [get_round_path(output_path, i) for i in existing_rounds]
    if ensemble_model_names:
        ensemble_model = ModelEnsemble(
            ensemble_model_names, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            vocab_size=student_model.vocab_size
        )
        ensemble_model.requires_grad_(False)
        print(f"Loaded ensemble with {len(ensemble_model_names)} models from rounds: {existing_rounds}")
        
        # Evaluate the current ensemble
        ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], ensemble_model.device, collator)
        print(f"Current ensemble evaluation: {ensemble_eval_results}")
        wandb.log({
            "ensemble/eval_loss": ensemble_eval_results["eval_loss"],
            "ensemble/size": len(ensemble_model.models),
            "current_round": -1  # Indicate this is evaluation
        })
        
    else:
        ensemble_model = None
        print("No prior ensemble loaded")

    # Start training from the next round
    start_round = max(existing_rounds) + 1 if existing_rounds else 0

    # Store all evaluation results for final comparison
    all_student_results = {}

    for round_num in range(start_round, config.total_rounds):
        cleanup_memory()
        
        # Record round start time
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*50}")
        print(f"Starting Round {round_num} at: {round_start_datetime}")
        print(f"{'='*50}")
        
        round_output_dir = get_round_path(output_path, round_num)
    
        # Shuffle the dataset with a different seed each round
        dataset["train"] = dataset["train"].shuffle(seed=config.seed+round_num)
        
        # Callback to log under the round-specific namespace
        class RoundSpecificCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    if 'loss' in logs and 'eval_loss' not in logs: # save only the training loss
                        round_logs = {f"round_{round_num}/train/{k}": v for k, v in logs.items()}
                        # Include the round number so we can plot by round
                        round_logs["round"] = round_num
                        wandb.log(round_logs, step=state.global_step)
        
        # Get training arguments
        training_args = config.get_training_args(round_output_dir)
         
        # Create the trainer
        trainer = DistillationTrainer(
            round_num=round_num,
            steps_per_round=config.steps_per_round,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[RoundSpecificCallback],
        )   
    
        # debug: log out the vocab size
        print(f"Model {i} vocab size: {model.get_input_embeddings().weight.shape[0]}")
        print(f"Student model vocab size: {student_model.get_input_embeddings().weight.shape[0]}")
        
        # Train the model
        trainer.train()
        
        # Evaluate the student model
        student_eval_results = evaluate_model(trainer.model, dataset["test"], trainer.model.device, collator)
        print(f"Round {round_num} student evaluation: {student_eval_results}")
        
        # Store results for this student model
        all_student_results[round_num] = student_eval_results

        # Save the model
        student_model.save_pretrained(round_output_dir)
        tokenizer.save_pretrained(round_output_dir)

        # Add the model to the ensemble
        if ensemble_model is None:
            ensemble_model = ModelEnsemble([round_output_dir], 
                            torch_dtype=torch.bfloat16, device_map="auto", 
                            vocab_size=student_model.vocab_size)
            ensemble_model.requires_grad_(False)
        else:
            ensemble_model.add_model(round_output_dir)
        
        if ensemble_model is not None:
            print(f"Ensemble has {len(ensemble_model.models)} models")
            for i, model in enumerate(ensemble_model.models):
                print(f"Model {i} vocab size: {model.get_input_embeddings().weight.shape[0]}")
                # Make sure all models have same vocab size
                # debug
                
                if model.get_input_embeddings().weight.shape[0] != student_model.get_input_embeddings().weight.shape[0]:
                    print(f"WARNING: Vocab size mismatch detected! Fixing...")
                    model.resize_token_embeddings(student_model.get_input_embeddings().weight.shape[0])
        
        # Evaluate the updated ensemble
        ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], ensemble_model.device, collator)
        print(f"Ensemble evaluation after round {round_num}: {ensemble_eval_results}")
        
        # Log all metrics in a consistent structure
        metrics = {
            "round": round_num, # Round number for X-axis
            "student/eval_loss": student_eval_results["eval_loss"], # Student eval metrics (current round)
            "ensemble/eval_loss": ensemble_eval_results["eval_loss"], # Ensemble metrics
            "ensemble/size": len(ensemble_model.models),
            # New metrics:
            "performance/teacher_vs_ensemble_gap": teacher_eval_results["eval_loss"] - ensemble_eval_results["eval_loss"],
            "performance/ensemble_improvement": previous_ensemble_loss - ensemble_eval_results["eval_loss"] if round_num > 0 else 0,
            "performance/teacher_achievement_pct": (teacher_eval_results["eval_loss"] / ensemble_eval_results["eval_loss"]) * 100,
        }
        wandb.log(metrics)
        
        # Track previous ensemble loss for improvement calculation
        previous_ensemble_loss = ensemble_eval_results["eval_loss"]
        
        # After training, record round end time
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        round_duration_str = format_time_elapsed(round_duration)
        
        # Calculate overall time elapsed so far
        overall_elapsed = round_end_time - overall_start_time
        overall_elapsed_str = format_time_elapsed(overall_elapsed)
        
        print(f"Round {round_num} completed in: {round_duration_str}")
        print(f"Total training time so far: {overall_elapsed_str}")
        
        # Log timing information to wandb
        timing_metrics = {
            "time/round_duration_seconds": round_duration,
            "time/round_duration_minutes": round_duration / 60.0,
            "time/total_elapsed_seconds": overall_elapsed,
            "time/total_elapsed_minutes": overall_elapsed / 60.0,
            "time/round": round_num
        }
        wandb.log(timing_metrics) 
        
        # Reset the student model for the next round
        del student_model
        cleanup_memory()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        student_model = AutoModelForCausalLM.from_pretrained( # Loads a fresh copy of the student base model
            config.student_model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        student_model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size)
        
    # Log final metrics
    student_table = wandb.Table(columns=["Round", "Eval Loss"])
    for round_num, results in all_student_results.items():
        student_table.add_data(round_num, results["eval_loss"])
    wandb.log({"student_performance_table": student_table})  
    
    # Record overall end time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_str = format_time_elapsed(overall_duration)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*50}")
    print(f"Training completed at: {end_datetime}")
    print(f"Total training time: {overall_duration_str}")
    print(f"{'='*50}")
    
    # Log final timing information
    final_timing = {
        "time/total_training_seconds": overall_duration,
        "time/total_training_minutes": overall_duration / 60.0,
        "time/average_round_minutes": (overall_duration / 60.0) / (config.total_rounds - start_round),
        "time/training_end_time": end_datetime
    }
    wandb.log(final_timing)
    
    # Create timing summary table
    timing_table = wandb.Table(columns=["Round", "Duration (min)", "Cumulative (min)"])
    total_mins = 0
    for r in range(start_round, config.total_rounds):
        # Get the round duration from our logs if available
        round_duration_min = wandb.run.summary.get(f"time/round_duration_minutes_{r}", 0)
        total_mins += round_duration_min
        timing_table.add_data(r, round_duration_min, total_mins)
    
    wandb.log({"time/summary_table": timing_table})
    
    # Close wandb
    wandb.finish() 
    

if __name__ == "__main__":
    main()
