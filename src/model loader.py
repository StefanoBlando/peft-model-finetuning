"""
Model and dataset loading utilities for the PEFT project.
"""

import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple, Optional, Dict, Any


class ModelLoader:
    """Handles loading of models and datasets for PEFT experiments."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        
    def load_model(self, num_labels: int = 2) -> Tuple[Any, Any]:
        """
        Load the base model and tokenizer.
        
        Args:
            num_labels: Number of classification labels
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Tokenizer loaded with vocabulary size: {len(self.tokenizer)}")
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model, self.tokenizer
    
    def load_sst2_dataset(self, train_size: int = 5000, val_size: int = 500) -> Tuple[Any, Any, Any]:
        """
        Load the SST-2 dataset with multiple fallback methods.
        
        Args:
            train_size: Number of training examples to use
            val_size: Number of validation examples for quick evaluation
            
        Returns:
            Tuple of (train_dataset, validation_dataset, full_validation_dataset)
        """
        print("Loading SST-2 dataset...")
        
        # Method 1: Standard loading
        try:
            dataset = load_dataset("glue", "sst2")
            print("Dataset loaded successfully!")
        except Exception as e:
            print(f"Error in standard loading: {e}")
            # Method 2: Manual download fallback
            dataset = self._manual_download_sst2()
        
        # Dataset statistics
        print(f"Training examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        
        # Create subsets
        train_dataset = dataset['train'].shuffle(seed=42).select(range(min(train_size, len(dataset['train']))))
        validation_subset = dataset['validation'].shuffle(seed=42).select(range(min(val_size, len(dataset['validation']))))
        full_validation = dataset['validation']
        
        print(f"Using {len(train_dataset)} training examples")
        print(f"Using {len(validation_subset)} validation examples for quick evaluations")
        
        # Preprocess datasets
        train_encoded = self._preprocess_dataset(train_dataset)
        val_encoded = self._preprocess_dataset(validation_subset)
        full_val_encoded = self._preprocess_dataset(full_validation)
        
        return train_encoded, val_encoded, full_val_encoded
    
    def _manual_download_sst2(self) -> DatasetDict:
        """Fallback method to manually download SST-2 data."""
        print("Attempting manual download of SST-2...")
        
        try:
            # Download if not present
            if not os.path.exists('SST-2.zip'):
                os.system('wget -q https://dl.fbaipublicfiles.com/glue/data/SST-2.zip')
                os.system('unzip -q SST-2.zip')
            elif not os.path.exists('SST-2'):
                os.system('unzip -q SST-2.zip')
            
            # Load data
            train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
            dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')
            
            # Rename columns if needed
            if 'sentence' not in train_df.columns and 'text' in train_df.columns:
                train_df = train_df.rename(columns={'text': 'sentence'})
                dev_df = dev_df.rename(columns={'text': 'sentence'})
            
            # Convert to datasets format
            train_dataset = Dataset.from_pandas(train_df)
            validation_dataset = Dataset.from_pandas(dev_df)
            
            dataset = DatasetDict({
                'train': train_dataset,
                'validation': validation_dataset
            })
            
            print("Dataset loaded successfully via manual download!")
            return dataset
            
        except Exception as e:
            raise ValueError(f"All dataset loading methods failed: {e}")
    
    def _preprocess_dataset(self, dataset) -> Any:
        """
        Preprocess dataset by tokenizing and formatting for PyTorch.
        
        Args:
            dataset: Raw dataset to preprocess
            
        Returns:
            Preprocessed dataset
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        def preprocess_function(examples):
            return self.tokenizer(
                examples["sentence"] if "sentence" in examples else examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
        
        # Tokenize
        encoded_dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Tokenizing dataset"
        )
        
        # Format for PyTorch
        encoded_dataset = self._format_for_pytorch(encoded_dataset)
        
        return encoded_dataset
    
    def _format_for_pytorch(self, dataset) -> Any:
        """Format dataset for PyTorch training."""
        # Remove unnecessary columns
        columns_to_remove = []
        if "sentence" in dataset.column_names:
            columns_to_remove.append("sentence")
        if "idx" in dataset.column_names:
            columns_to_remove.append("idx")
        
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Rename label to labels for Trainer compatibility
        if "label" in dataset.column_names and "labels" not in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        
        # Set format to PyTorch tensors
        dataset.set_format("torch")
        
        return dataset
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            "model_type": model.config.model_type if hasattr(model, 'config') else "Unknown",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
            "model_size_mb": total_size_mb,
            "model_size_str": f"{total_size_mb/1024:.2f} GB" if total_size_mb > 1024 else f"{total_size_mb:.2f} MB"
        }
    
    def analyze_dataset(self, dataset) -> Dict[str, Any]:
        """
        Analyze dataset characteristics.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        if 'labels' in dataset.features or 'label' in dataset.features:
            label_col = 'labels' if 'labels' in dataset.features else 'label'
            labels = dataset[label_col]
            label_counts = pd.Series(labels).value_counts().to_dict()
            
            return {
                "num_examples": len(dataset),
                "label_distribution": label_counts,
                "num_classes": len(label_counts),
                "class_balance": {
                    label: count / len(dataset) * 100 
                    for label, count in label_counts.items()
                }
            }
        else:
            return {
                "num_examples": len(dataset),
                "features": list(dataset.features.keys())
            }
