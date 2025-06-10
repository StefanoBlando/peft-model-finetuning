"""
QLoRA (Quantized LoRA) training implementation for extreme memory efficiency.
"""

import os
import time
import torch
from typing import Dict, Any, Optional, List
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


class QLoRATrainer:
    """Handles QLoRA (Quantized LoRA) training and evaluation."""
    
    def __init__(self, model_name: str, tokenizer, device=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = []
        
    def check_qlora_requirements(self) -> bool:
        """
        Check if QLoRA requirements are available.
        
        Returns:
            True if QLoRA can be used, False otherwise
        """
        try:
            import bitsandbytes as bnb
            print("bitsandbytes is available for QLoRA")
            return True
        except ImportError:
            print("bitsandbytes not available. QLoRA will not work.")
            return False
    
    def load_quantized_model(
        self,
        num_labels: int = 2,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False
    ):
        """
        Load model with quantization for QLoRA.
        
        Args:
            num_labels: Number of classification labels
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            
        Returns:
            Quantized model ready for QLoRA
        """
        if not self.check_qlora_requirements():
            raise ImportError("bitsandbytes is required for QLoRA but not available")
        
        print(f"Loading {self.model_name} with quantization...")
        
        # Load model with quantization
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            load_in_8bit=load_in_8bit if not load_in_4bit else False,
            load_in_4bit=load_in_4bit,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        print("Model loaded and prepared for QLoRA training")
        return model
    
    def create_qlora_config(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none"
    ) -> LoraConfig:
        """
        Create QLoRA configuration with conservative settings.
        
        Args:
            r: Rank of the update matrices (lower for QLoRA)
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout probability (lower for stability)
            target_modules: List of modules to target
            bias: Bias training strategy
            
        Returns:
            LoRA configuration optimized for QLoRA
        """
        if target_modules is None:
            # Conservative target modules for stability
            target_modules = ["q_lin", "v_lin"]
        
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            inference_mode=False,
        )
    
    def setup_qlora_model(self, quantized_model, qlora_config: LoraConfig):
        """
        Setup QLoRA model with adapters.
        
        Args:
            quantized_model: Quantized base model
            qlora_config: QLoRA configuration
            
        Returns:
            QLoRA model with adapters
        """
        # Create PEFT model
        qlora_model = get_peft_model(quantized_model, qlora_config)
        
        # Print parameter information
        trainable_params = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in qlora_model.parameters())
        percentage = 100 * trainable_params / total_params
        
        print(f"QLoRA trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage of parameters being trained: {percentage:.4f}%")
        print(f"Parameter reduction factor: {total_params / trainable_params:.2f}x")
        
        return qlora_model
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./qlora_model",
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 8,
        eval_batch_size: int = 16,
        weight_decay: float = 0.0,
        logging_steps: int = 50,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        qlora_config: Optional[LoraConfig] = None,
        quantization_bits: int = 8,
        **kwargs
    ):
        """
        Train the QLoRA model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model outputs
            learning_rate: Learning rate (lower for QLoRA stability)
            num_epochs: Number of training epochs
            batch_size: Training batch size (smaller for memory)
            eval_batch_size: Evaluation batch size
            weight_decay: Weight decay (0 for stability)
            logging_steps: Steps between logging
            save_strategy: When to save checkpoints
            evaluation_strategy: When to evaluate
            load_best_model_at_end: Whether to load best model at end
            qlora_config: QLoRA configuration
            quantization_bits: Bits for quantization (4 or 8)
            **kwargs: Additional training arguments
            
        Returns:
            Trained QLoRA model and trainer
        """
        # Load quantized model
        quantized_model = self.load_quantized_model(
            load_in_8bit=(quantization_bits == 8),
            load_in_4bit=(quantization_bits == 4)
        )
        
        # Create QLoRA config if not provided
        if qlora_config is None:
            qlora_config = self.create_qlora_config()
        
        # Setup QLoRA model
        qlora_model = self.setup_qlora_model(quantized_model, qlora_config)
        
        # Training arguments optimized for QLoRA
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            logging_dir=f"{output_dir}/logs",
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",
            # Critical: Disable mixed precision for quantized models
            fp16=False,
            bf16=False,
            gradient_accumulation_steps=2,  # Help with smaller batch sizes
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=qlora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train the model
        print(f"Starting QLoRA training...")
        print(f"Training on {len(train_dataset)} examples")
        print(f"Evaluating on {len(eval_dataset)} examples")
        print(f"Using {quantization_bits}-bit quantization")
        
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"QLoRA training completed in {self._format_time(training_time)}")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save the model
        qlora_model.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "training_time_seconds": training_time,
            "training_loss": train_result.training_loss,
            "quantization_bits": quantization_bits,
            "qlora_config": {
                "r": qlora_config.r,
                "lora_alpha": qlora_config.lora_alpha,
                "lora_dropout": qlora_config.lora_dropout,
                "target_modules": qlora_config.target_modules,
                "bias": qlora_config.bias,
            },
            "training_args": {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            }
        }
        
        with open(f"{output_dir}/qlora_training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Store training history
        self.training_history = trainer.state.log_history if hasattr(trainer, 'state') else []
        
        return qlora_model, trainer
    
    def evaluate(self, model, eval_dataset, batch_size: int = 16) -> Dict[str, Any]:
        """
        Evaluate a QLoRA model on a dataset.
        
        Args:
            model: QLoRA model to evaluate
            eval_dataset: Dataset for evaluation
            batch_size: Evaluation batch size
            
        Returns:
            Dictionary of evaluation metrics
        """
        eval_args = TrainingArguments(
            output_dir="./eval_tmp",
            per_device_eval_batch_size=batch_size,
            do_train=False,
            do_eval=True,
            report_to="none",
            fp16=False,  # Important for quantized models
            bf16=False
        )
        
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        return trainer.evaluate()
    
    def calculate_memory_savings(self, base_memory_mb: float) -> Dict[str, Any]:
        """
        Calculate memory savings compared to base model.
        
        Args:
            base_memory_mb: Base model memory usage in MB
            
        Returns:
            Dictionary with memory statistics
        """
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            # Rough estimate for CPU
            current_memory = base_memory_mb * 0.6  # Assume ~40% reduction
        
        savings = base_memory_mb - current_memory
        savings_pct = (savings / base_memory_mb) * 100 if base_memory_mb > 0 else 0
        
        return {
            "base_memory_mb": base_memory_mb,
            "qlora_memory_mb": current_memory,
            "memory_savings_mb": savings,
            "memory_savings_percent": savings_pct
        }
    
    def compare_quantization_levels(
        self,
        train_dataset,
        eval_dataset,
        output_base_dir: str = "./qlora_comparison"
    ) -> Dict[str, Any]:
        """
        Compare different quantization levels (4-bit vs 8-bit).
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_base_dir: Base directory for outputs
            
        Returns:
            Comparison results
        """
        results = {}
        
        for bits in [8, 4]:
            print(f"\nTesting {bits}-bit quantization...")
            
            try:
                output_dir = f"{output_base_dir}/qlora_{bits}bit"
                os.makedirs(output_dir, exist_ok=True)
                
                # Train with current quantization level
                model, trainer = self.train(
                    train_dataset=train_dataset.select(range(min(1000, len(train_dataset)))),
                    eval_dataset=eval_dataset,
                    output_dir=output_dir,
                    num_epochs=1,  # Quick comparison
                    quantization_bits=bits,
                    save_strategy="no"
                )
                
                # Evaluate
                eval_results = self.evaluate(model, eval_dataset)
                
                # Calculate memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                else:
                    memory_mb = 400 if bits == 8 else 300  # Rough estimates
                
                results[f"{bits}bit"] = {
                    "quantization_bits": bits,
                    "memory_usage_mb": memory_mb,
                    **{k.replace('eval_', ''): v for k, v in eval_results.items() 
                       if k.startswith('eval_') and not k.startswith('eval_runtime')}
                }
                
                print(f"{bits}-bit results: Accuracy={results[f'{bits}bit'].get('accuracy', 0):.4f}, "
                      f"Memory={memory_mb:.1f}MB")
                
            except Exception as e:
                print(f"Error with {bits}-bit quantization: {e}")
                continue
        
        return results
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def get_training_history(self) -> List[Dict]:
        """Get the training history from the last training run."""
        return self.training_history
    
    def predict_batch(self, model, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts using QLoRA model.
        
        Args:
            model: Trained QLoRA model
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            label = "Positive" if pred == 1 else "Negative"
            confidence = probabilities[i][pred].item()
            
            results.append({
                "text": texts[i],
                "prediction": int(pred),
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "Negative": probabilities[i][0].item(),
                    "Positive": probabilities[i][1].item()
                }
            })
        
        return results
