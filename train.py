import os
import time
import wandb
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import speed_metrics  # Added this import
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  # Fixed SFTTrainer import

from ensemble import ModelEnsemble
import config

teacher_model = None
ensemble_model = None

def get_round_path(output_path, round_num):
    """Return path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")

def evaluate_model(model, eval_dataset, tokenizer, device, collate_fn):
    """Evaluates a model and returns the language modeling loss."""
    model.eval()
    total_loss = 0
    num_steps = 0

    # Create a DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_steps += 1

    avg_loss = total_loss / num_steps
    return {"eval_loss": avg_loss}


class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.training_round = kwargs.pop("training_round")
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
        
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids.to(teacher_model.device), attention_mask=attention_mask.detach().to(teacher_model.device)).logits
            if ensemble_model is not None:
                ensemble_logits = ensemble_model(input_ids.to(ensemble_model.device), attention_mask=attention_mask.detach().to(ensemble_model.device)).logits
            else:
                ensemble_logits = None
        
        student_logits = model(input_ids, attention_mask=attention_mask).logits
        loss = self.compute_kl_loss(student_logits, ensemble_logits, teacher_logits, labels != -100) # -100 = only include tokens that have valid labels
        return (loss, student_logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            student_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            if ensemble_model is not None:
                num_models = len(ensemble_model.models)
                ensemble_logits = ensemble_model(input_ids=inputs["input_ids"].to(ensemble_model.device), attention_mask=inputs["attention_mask"].to(ensemble_model.device)).logits.to(student_logits.device)
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
                loss = torch.tensor(0.0).to(student_logits.device)

        return (loss, None, None) if prediction_loss_only else (loss, student_logits, inputs["labels"])
    
    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step + (self.training_round*self.steps_per_round)}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

# Define a callback to save evaluation metrics to wandb
class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)

def main():
    global teacher_model, ensemble_model

    # Get run directory
    output_path = config.get_run_directory()
    print(f"Models stored in: {output_path}")

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    global teacher_model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    teacher_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
    teacher_model.requires_grad_(False)

    # Initialize ensemble model
    global ensemble_model
    if len(config.ensemble_model_names) > 0:
        ensemble_model = ModelEnsemble(
            config.ensemble_model_names, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            vocab_size=student_model.vocab_size
        )
        ensemble_model.requires_grad_(False)
    else:
        ensemble_model = None

    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Evaluate the teacher model
    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], tokenizer, teacher_model.device, collator)
    print(f"Teacher model evaluation: {teacher_eval_results}")

    # Setup checkpoint saving
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check for existing rounds and setup ensemble
    existing_rounds = []
    for i in range(config.total_rounds):
        round_path = os.path.join(output_path, f"round_{i}")
        if os.path.exists(round_path):
            existing_rounds.append(i)
            
    # If we have existing rounds, load the ensemble
    ensemble_model_names = [os.path.join(output_path, f"round_{i}") for i in existing_rounds]
    if ensemble_model_names:
        ensemble_model = ModelEnsemble(
            ensemble_model_names, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            vocab_size=student_model.vocab_size
        )
        ensemble_model.requires_grad_(False)
        print(f"Loaded ensemble with {len(ensemble_model_names)} models from rounds: {existing_rounds}")
    else:
        ensemble_model = None
        print("Starting from scratch with no ensemble")

    # Start training from the next round
    start_round = max(existing_rounds) + 1 if existing_rounds else 0

    for training_round in range(start_round, config.total_rounds):
        round_output_dir = get_round_path(output_path, training_round)

        # Setup wandb
        run_name = f"{os.path.basename(output_path)}_round_{training_round}"
        wandb.init(project="slm_ensembles", name=run_name, reinit=True)
        wandb.config.update({
            "round": training_round,
            "student_model": config.student_model_name,
            "teacher_model": config.teacher_model_name,
            "ensemble_size": len(ensemble_model.models) if ensemble_model else 0,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "steps_per_round": config.steps_per_round
        })
    
        # Shuffle the dataset with a different seed each round
        dataset["train"] = dataset["train"].shuffle(seed=config.seed+training_round)
        
        # Get training arguments
        training_args = config.get_training_args(round_output_dir)
         
        # Create the trainer
        trainer = DistillationTrainer(
            training_round=training_round,
            steps_per_round=config.steps_per_round,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[WandbCallback],
        )   
    
        # Train the model
        trainer.train()
        
        # Evaluate the student model
        student_eval_results = evaluate_model(trainer.model, dataset["test"], tokenizer, trainer.model.device, collator)
        print(f"Round {training_round} student evaluation: {student_eval_results}")
        
        # Log metrics
        log_data = {
            "round": training_round,
            "student_eval_loss": student_eval_results["eval_loss"],
            "ensemble_size": len(ensemble_model.models) if ensemble_model else 0,
        }
        wandb.log(log_data)

            # Save the model
        student_model.save_pretrained(os.path.join(output_path, f"round_{training_round}"))
        tokenizer.save_pretrained(os.path.join(output_path, f"round_{training_round}"))

        # Add the model to the ensemble
        if ensemble_model is None:
            ensemble_model = ModelEnsemble([os.path.join(output_path, f"round_{training_round}")], 
                            torch_dtype=torch.bfloat16, device_map="auto", 
                            vocab_size=student_model.vocab_size)
            ensemble_model.requires_grad_(False)
        else:
            ensemble_model.add_model(os.path.join(output_path, f"round_{training_round}"))
        
        # Reset the student model for the next round
        del student_model
        torch.cuda.empty_cache()  # Clear CUDA cache to avoid memory issues
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        student_model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size)
        
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()
