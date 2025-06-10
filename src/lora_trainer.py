"""
LoRA training implementation for parameter-efficient fine-tuning.
"""

import os
import time
import torch
from typing import Dict, Any, Optional, List
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


class LoRATrainer:
    """Handles LoRA (Low-Rank Adaptation) training and evaluation."""
    
    def __init__(self, base_model, tokenizer, device=None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = []
        
    def create_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none"
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            r: Rank of the update matrices
            lora_alpha: Scaling factor for LoRA
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to target (None for auto-detection)
            bias: Bias training strategy
            
        Returns:
            LoRA configuration object
        """
        if target_modules is None:
            # Default target modules for DistilBERT
            target_modules = ["q_lin", "v_lin", "k_lin", "out_lin"]
        
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            inference_mode=False,
        )
    
    def setup_lora_model(self, lora_config: LoraConfig):
        """
        Setup the PEFT model with LoRA configuration.
        
        Args:
            lora_config: LoRA configuration
            
        Returns:
            PEFT model with LoRA adapters
        """
        # Create PEFT model
        peft_model = get_peft_model(self.base_model, lora_config)
        peft_model = peft_model.to(self.device)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        percentage = 100 * trainable_params / total_params
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage of parameters being trained: {percentage:.4f}%")
        print(f"Parameter reduction factor: {total_params / trainable_params:.2f}x")
        
        return peft_model
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./lora_model",
        learning_rate: float = 5e-4,
        num_epochs: int = 3,
        batch_size: int = 16,
        eval_batch_size: int = 32,
        weight_decay: float = 0.01,
        logging_steps: int = 50,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        lora_config: Optional[LoraConfig] = None,
        **kwargs
    ):
        """
        Train the LoRA model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model outputs
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            weight_decay: Weight decay for regularization
            logging_steps: Steps between logging
            save_strategy: When to save checkpoints
            evaluation_strategy: When to evaluate
            load_best_model_at_end: Whether to load best model at end
            lora_config: LoRA configuration (creates default if None)
            **kwargs: Additional training arguments
            
        Returns:
            Trained PEFT model
        """
        # Create LoRA config if not provided
        if lora_config is None:
            lora_config = self.create_lora_config()
        
        # Setup LoRA model
        peft_model = self.setup_lora_model(lora_config)
        
        # Training arguments
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
            fp16=torch.cuda.is_available(),
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train the model
        print(f"Starting LoRA training...")
        print(f"Training on {len(train_dataset)} examples")
        print(f"Evaluating on {len(eval_dataset)} examples")
        
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"Training completed in {self._format_time(training_time)}")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save the model
        peft_model.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "training_time_seconds": training_time,
            "training_loss": train_result.training_loss,
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules,
                "bias": lora_config.bias,
            },
            "training_args": {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            }
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Store training history
        self.training_history = trainer.state.log_history if hasattr(trainer, 'state') else []
        
        return peft_model, trainer
    
    def evaluate(self, model, eval_dataset, batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
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
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        return trainer.evaluate()
    
    def compare_configurations(
        self,
        train_dataset,
        eval_dataset,
        configurations: List[Dict[str, Any]],
        output_base_dir: str = "./lora_variants"
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple LoRA configurations.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            configurations: List of configuration dictionaries
            output_base_dir: Base directory for saving variants
            
        Returns:
            List of results for each configuration
        """
        results = []
        
        for i, config in enumerate(configurations):
            print(f"\nTesting configuration {i+1}/{len(configurations)}: {config['name']}")
            
            # Create LoRA config
            lora_config = self.create_lora_config(**config['params'])
            
            # Output directory for this variant
            variant_dir = f"{output_base_dir}/variant_{i}"
            os.makedirs(variant_dir, exist_ok=True)
            
            try:
                # Train model with quick settings (1 epoch for comparison)
                model, trainer = self.train(
                    train_dataset=train_dataset.select(range(min(2000, len(train_dataset)))),
                    eval_dataset=eval_dataset,
                    output_dir=variant_dir,
                    num_epochs=1,
                    lora_config=lora_config,
                    save_strategy="no"  # Don't save checkpoints
                )
                
                # Evaluate on full dataset
                eval_results = self.evaluate(model, eval_dataset)
                
                # Calculate parameter efficiency
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                
                result = {
                    "config_name": config['name'],
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "param_efficiency": trainable_params / total_params * 100,
                    **{k.replace('eval_', ''): v for k, v in eval_results.items() 
                       if k.startswith('eval_') and not k.startswith('eval_runtime')}
                }
                
                results.append(result)
                print(f"Results: Accuracy={result.get('accuracy', 0):.4f}, "
                      f"Parameters={trainable_params:,} ({result['param_efficiency']:.2f}%)")
                
            except Exception as e:
                print(f"Error training configuration {config['name']}: {e}")
                continue
        
        return results
    
    def get_default_configurations(self) -> List[Dict[str, Any]]:
        """
        Get a set of default LoRA configurations for comparison.
        
        Returns:
            List of configuration dictionaries
        """
        return [
            {
                "name": "Low Rank (r=4)",
                "params": {"r": 4, "lora_alpha": 16}
            },
            {
                "name": "Standard (r=16)",
                "params": {"r": 16, "lora_alpha": 32}
            },
            {
                "name": "High Rank (r=32)",
                "params": {"r": 32, "lora_alpha": 64}
            },
            {
                "name": "Query-Only",
                "params": {"r": 16, "lora_alpha": 32, "target_modules": ["q_lin"]}
            },
            {
                "name": "With Bias",
                "params": {"r": 16, "lora_alpha": 32, "bias": "lora_only"}
            },
            {
                "name": "High Dropout",
                "params": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.3}
            }
        ]
    
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
        Predict sentiment for a batch of texts.
        
        Args:
            model: Trained model
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
        ).to(self.device)
        
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
