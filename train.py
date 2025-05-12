import os
import time

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from transformers.trainer_utils import speed_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
# SFT Trainer = specialized trainer class from HuggingFace's TRL (Transformer Reinforcement Learning) library specifically designed for Supervised Fine-Tuning (SFT) of language models
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainerCallback
import numpy as np
import inspect

from ensemble import ModelEnsemble


# Model and dataset setup
seed = 42
dataset_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
output_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/boosted_distillation_1.5B_teacher_average_fixed_logging"
teacher_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
ensemble_model_names = []

tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16, device_map="auto")

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16, device_map="auto")
teacher_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
teacher_model.requires_grad_(False)

# initializes the ensemble model if ensemble model naems are provided
if len(ensemble_model_names) > 0:
    ensemble_model = ModelEnsemble(ensemble_model_names, torch_dtype=torch.bfloat16, device_map="auto", vocab_size=student_model.vocab_size)
    ensemble_model.requires_grad_(False)
else:
    ensemble_model = None
 
# loads the pre-tokenized dataset and sets up the data collator to focus training only on the assistant's responses
# data collator = processes batches of data in a specific way; pads the sequences
dataset = datasets.load_from_disk(dataset_path)
response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

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
        return (loss, student_outputs) if return_outputs else loss

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

# Initialize wandb
os.environ["WANDB_PROJECT"] = "slm_ensembles"
os.environ["WANDB_LOG_MODEL"] = "true"

# Setup checkpoint saving and resuming
checkpoint_dir = os.path.join(output_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Sets up the ensemble with models from previous training rounds
existing_rounds = []
for i in range(6):
    round_path = os.path.join(output_path, f"round_{i}")
    if os.path.exists(round_path):
        existing_rounds.append(i)

# If we have existing rounds, load the ensemble
ensemble_model_names = [os.path.join(output_path, f"round_{i}") for i in existing_rounds]
if ensemble_model_names:
    ensemble_model = ModelEnsemble(ensemble_model_names, torch_dtype=torch.bfloat16, device_map="auto", 
                                   vocab_size=student_model.vocab_size)
    ensemble_model.requires_grad_(False)
    print(f"Loaded ensemble with {len(ensemble_model_names)} models from rounds: {existing_rounds}")
else:
    ensemble_model = None
    print("Starting from scratch with no ensemble")

# Define a callback to save evaluation metrics to wandb
class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)

steps_per_round = 1000
start_round = max(existing_rounds) + 1 if existing_rounds else 0

for training_round in range(start_round,6):
    # Setup wandb for this round
    run_name = f"round_{training_round}"
    wandb.init(project="slm_ensembles", name=run_name, reinit=True)
    wandb.config.update({
        "round": training_round,
        "student_model": student_model_name,
        "teacher_model": teacher_model_name,
        "ensemble_size": len(ensemble_model.models) if ensemble_model else 0,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "steps_per_round": steps_per_round
    })
    
    # Shuffle the dataset with a different seed each round
    dataset["train"] = dataset["train"].shuffle(seed=seed+training_round)
    
    training_args = SFTConfig(
        output_dir=os.path.join(output_path, f"round_{training_round}"),
        overwrite_output_dir=True,
        report_to="wandb",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        warmup_steps=50,
        bf16=True,
        max_steps=steps_per_round,
        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
     # Create the trainer
    trainer = DistillationTrainer(
        training_round=training_round,
        steps_per_round=steps_per_round,
        model=student_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        args=training_args,
        tokenizer=tokenizer,  # Fixed: changed from processing_class to tokenizer
        callbacks=[WandbCallback],
    )
    
    # train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Round {training_round} evaluation results: {eval_results}")

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
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    student_model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size)
    
    # Close wandb for this round
    wandb.finish()
