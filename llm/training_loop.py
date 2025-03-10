import time
import os
import json
import torch
import numpy as np
import evaluate  # Hugging Face evaluation metrics
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Global paths for model and dataset
MODEL_PATH = "/Users/Downloads/llama-2-1.3b/"
DATA_PATH = "/Users/Downloads/alpaca_data.json"
BEST_SCORE_FILE = "./best_score.txt"
OUTPUT_DIR = "./llama2_results"
LOGGING_DIR = "./logs"

# IF A DIFFERENT DATASET IS USED, UPDATE THESE FUNCTIONS: load_alpaca_data(), tokenize_function()

# Load evaluation metric (accuracy)
metric = evaluate.load("accuracy")


class LLaMAFineTuner:
    def __init__(self, model_path=MODEL_PATH, data_path=DATA_PATH, max_length=512):
        self.model_path = model_path
        self.data_path = data_path
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Fix padding issue

        # Load model without `bitsandbytes`
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Use standard float32 precision
            device_map="cpu",
            local_files_only=True
        )

        # Apply LoRA fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Load and preprocess the dataset
        self.dataset = self.load_alpaca_data()

        # Define a data collator for padding and truncation
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=True)

        # Define training arguments optimized for CPU fine-tuning
        self.training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            gradient_accumulation_steps=32,
            fp16=False,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=LOGGING_DIR
        )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def load_alpaca_data(self):
        """Loads and formats the Alpaca dataset into a Hugging Face Dataset format."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            alpaca_data = json.load(f)

        formatted_data = []
        for example in alpaca_data:
            instruction = example["instruction"]
            input_text = example.get("input", "").strip()
            output_text = example["output"]

            # Construct the training prompt following the Alpaca dataset format
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:"

            formatted_data.append({"text": prompt, "label": output_text})

        # Convert the list of formatted examples into a Hugging Face Dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize the dataset for training
        return dataset.map(self.tokenize_function, batched=True)

    def tokenize_function(self, examples):
        """Tokenizes the dataset with input-output pairs for training."""
        inputs = self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
        outputs = self.tokenizer(examples["label"], padding="max_length", truncation=True, max_length=self.max_length)
        inputs["labels"] = outputs["input_ids"]
        return inputs

    def compute_metrics(self, eval_pred):
        """
        Computes evaluation metrics for the model.
        
        Metrics included:
        - Loss: Directly obtained from the Hugging Face Trainer.
        - Perplexity (PPL): Exponential of the loss (lower is better).
        - Accuracy: Based on token-level correctness.

        Args:
            eval_pred: Tuple of (logits, labels) from evaluation.

        Returns:
            Dictionary with 'eval_loss', 'perplexity', and 'accuracy'.
        """
        logits, labels = eval_pred

        # Convert logits to probabilities and get predicted tokens
        predictions = np.argmax(logits, axis=-1)

        # Mask out padding tokens in labels
        valid_labels = labels != -100  # -100 is used as the ignore_index in Hugging Face datasets

        # Compute accuracy
        accuracy = metric.compute(predictions=predictions[valid_labels], references=labels[valid_labels])["accuracy"]

        # Compute loss (cross-entropy)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)

        with torch.no_grad():
            loss = loss_fct(logits_tensor.view(-1, logits_tensor.shape[-1]), labels_tensor.view(-1)).item()

        # Compute perplexity (lower perplexity means better predictions)
        perplexity = np.exp(loss) if loss < 20 else float("inf")  # Avoid overflow for large loss values

        return {
            "eval_loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }

    def train(self):
        """Runs the training process and records training time."""
        start_time = time.time()
        print("Starting LLaMA 2 fine-tuning on CPU...")
        self.trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training complete. Total time: {total_time:.2f} seconds.")
        return total_time

    def evaluate_and_save(self):
        """Evaluates the trained model and saves it if it achieves a better score."""
        eval_metrics = self.trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        current_score = 1.0 / (eval_loss + 1e-8)
        print(f"Evaluation loss: {eval_loss:.4f}, Perplexity: {eval_metrics['perplexity']:.2f}, Accuracy: {eval_metrics['accuracy']:.2f}")

        best_score = -float("inf")

        # Check if a best score exists from previous runs
        if os.path.exists(BEST_SCORE_FILE):
            with open(BEST_SCORE_FILE, "r") as f:
                try:
                    best_score = float(f.read().strip())
                except Exception:
                    pass

        # Save the model only if the current score is better
        if current_score > best_score:
            self.trainer.save_model()
            with open(BEST_SCORE_FILE, "w") as f:
                f.write(str(current_score))
            print(f"Model saved. New best score: {current_score:.4f}")
        else:
            print(f"Model not saved. Current score: {current_score:.4f} is not higher than best score: {best_score:.4f}")

    def run_all(self):
        """Runs the full training pipeline including training, evaluation, and saving the best model."""
        self.train()
        self.evaluate_and_save()

if __name__ == "__main__":
    tuner = LLaMAFineTuner()
    tuner.run_all()
