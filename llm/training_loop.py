import time
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

class LLMFineTuner:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", dataset_name="imdb", split="train[:1%]", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer and model in 8-bit mode for memory efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Apply LoRA fine-tuning configuration
        lora_config = LoraConfig(
            r=8,                   # LoRA rank
            lora_alpha=32,         # Scaling factor
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Load and preprocess a dataset (using IMDb for demonstration)
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)
        
        # Create a data collator for causal language modeling
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        # Define training arguments with advanced features
        self.training_args = TrainingArguments(
            output_dir="./advanced_results",
            evaluation_strategy="steps",       # Evaluate every eval_steps
            eval_steps=50,                     # Run evaluation every 50 training steps
            logging_steps=50,                  # Log training progress every 50 steps
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            gradient_accumulation_steps=4,     # Accumulate gradients over several steps
            fp16=True,                         # Use 16-bit precision
            save_total_limit=2,                # Limit total saved checkpoints
            load_best_model_at_end=True,       # Load best model based on evaluation metric
            metric_for_best_model="eval_loss", # Using eval_loss for checkpointing
            greater_is_better=False,           # Lower eval_loss is better
            logging_dir="./logs"
        )
        
        # Initialize the Trainer with an EarlyStopping callback
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset,   # Ideally, use a separate validation set
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # File to persist the best score across training runs
        self.best_score_file = "./best_score.txt"
    
    def tokenize_function(self, examples):
        # Tokenize and pad/truncate inputs to a fixed length
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
    
    def compute_metrics(self, eval_pred):
        # Placeholder for additional metric computations; here we rely on eval_loss only.
        return {}
    
    def train(self):
        start_time = time.time()
        print("Starting advanced LLM training...")
        self.trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training complete. Total training time: {total_time:.2f} seconds.")
        return total_time
    
    def evaluate_and_save(self):
        # Evaluate the model and compute a score (the lower the loss, the higher the score)
        eval_metrics = self.trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        epsilon = 1e-8
        current_score = 1.0 / (eval_loss + epsilon)
        print(f"Evaluation loss: {eval_loss:.4f}, Computed Score: {current_score:.4f}")
        
        # Check if there's a saved best score from previous runs
        best_score = -float("inf")
        if os.path.exists(self.best_score_file):
            with open(self.best_score_file, "r") as f:
                try:
                    best_score = float(f.read().strip())
                except Exception:
                    best_score = -float("inf")
        
        # Save the model only if the current score is higher than the best score so far
        if current_score > best_score:
            self.trainer.save_model()  # Save the fine-tuned model
            with open(self.best_score_file, "w") as f:
                f.write(str(current_score))
            print(f"Model saved. New best score: {current_score:.4f}")
        else:
            print(f"Model not saved. Current score: {current_score:.4f} is not higher than best score: {best_score:.4f}")
        return current_score

    def run_all(self):
        """
        Runs the full pipeline: training, evaluation, and conditional saving.
        Returns:
            total_time (float): The total training time.
            current_score (float): The evaluation score computed.
        """
        total_time = self.train()
        current_score = self.evaluate_and_save()
        return total_time, current_score

if __name__ == "__main__":
    tuner = LLMFineTuner()
    tuner.run_all()
