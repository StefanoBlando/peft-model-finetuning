"""Model and dataset loading utilities."""

import os
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed
)


class ModelLoader:
    """Handles loading of models, tokenizers, and datasets."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", seed: int = 42):
        """Initialize the model loader.
        
        Args:
            model_name: Name of the pre-trained model to load
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seeds for reproducibility
        set_seed(seed)
        torch.manual_seed(seed)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    
    def load_model(self, num_labels: int = 2) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load the base model and tokenizer.
        
        Args:
            num_labels: Number of labels for classification
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded with {total_params:,} parameters")
        print(f"Model size: {self._get_model_size(model)}")
        
        return model, tokenizer
    
    def load_sst2_dataset(self, train_size: int = 5000, val_size: int = 500) -> Tuple[Dataset, Dataset]:
        """Load and preprocess the SST-2 dataset.
        
        Args:
            train_size: Number of training examples to use
            val_size: Number of validation examples for quick evaluation
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        print("Loading SST-2 dataset...")
        
        try:
            # Load dataset with multiple fallback methods
            dataset = self._load_sst2_with_fallback()
            
            print(f"Dataset loaded with {len(dataset['train'])} training examples")
            print(f"Using {train_size} training examples")
            print(f"Using {val_size} validation examples")
            
            # Create subsets
            train_dataset = dataset['train'].shuffle(seed=self.seed).select(range(train_size))
            val_dataset = dataset['validation'].shuffle(seed=self.seed).select(range(val_size))
            
            # Check class balance
            self._analyze_class_distribution(dataset['train'])
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"Error loading SST-2 dataset: {e}")
            raise RuntimeError("Failed to load dataset")
    
    def preprocess_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer, 
                          max_length: int = 128) -> Dataset:
        """Preprocess dataset with tokenization.
        
        Args:
            dataset: Dataset to preprocess
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            """Tokenize and prepare examples."""
            return tokenizer(
                examples["sentence"] if "sentence" in examples else examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        print("Preprocessing dataset...")
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Tokenizing"
        )
        
        # Format for PyTorch
        processed_dataset = self._format_for_pytorch(processed_dataset)
        
        return processed_dataset
    
    def _load_sst2_with_fallback(self) -> DatasetDict:
        """Load SST-2 dataset with multiple fallback methods."""
        # Method 1: Standard loading
        try:
            print("Attempting to load SST-2 dataset directly...")
            dataset = load_dataset("glue", "sst2")
            print("Dataset loaded successfully!")
            return dataset
        except Exception as e:
            print(f"Standard loading failed: {e}")
        
        # Method 2: Try with dataset builder
        try:
            print("Attempting to load with dataset builder...")
            from datasets import load_dataset_builder
            builder = load_dataset_builder("glue", "sst2")
            builder.download_and_prepare()
            dataset = builder.as_dataset()
            print("Dataset loaded successfully with builder!")
            return dataset
        except Exception as e:
            print(f"Dataset builder failed: {e}")
        
        # Method 3: Manual download
        try:
            print("Attempting direct download of SST-2...")
            return self._manual_sst2_download()
        except Exception as e:
            print(f"Manual download failed: {e}")
            raise ValueError("All dataset loading methods failed")
    
    def _manual_sst2_download(self) -> DatasetDict:
        """Manually download and process SST-2 dataset."""
        import urllib.request
        import zipfile
        
        # Download if not exists
        if not os.path.exists('SST-2.zip'):
            print("Downloading SST-2 dataset...")
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
                'SST-2.zip'
            )
        
        # Extract if not exists
        if not os.path.exists('SST-2'):
            with zipfile.ZipFile('SST-2.zip', 'r') as zip_ref:
                zip_ref.extractall()
        
        # Load train and dev data
        train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
        dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')
        
        # Rename columns if needed
        if 'sentence' not in train_df.columns:
            train_df = train_df.rename(columns={'text': 'sentence'})
            dev_df = dev_df.rename(columns={'text': 'sentence'})
        
        # Convert to datasets format
        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(dev_df)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
    
    def _format_for_pytorch(self, dataset: Dataset) -> Dataset:
        """Format dataset for PyTorch."""
        # Remove unnecessary columns
        columns_to_remove = []
        for col in dataset.column_names:
            if col not in ['input_ids', 'attention_mask', 'label', 'labels']:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        
        # Rename label to labels if necessary
        if "label" in dataset.column_names and "labels" not in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        
        # Set format to PyTorch tensors
        dataset.set_format("torch")
        return dataset
    
    def _analyze_class_distribution(self, dataset: Dataset) -> None:
        """Analyze and print class distribution."""
        if 'label' in dataset.features:
            labels = dataset['label']
            label_counts = pd.Series(labels).value_counts()
            
            print("\nClass distribution in training set:")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count} examples ({count/len(labels)*100:.1f}%)")
    
    def _get_model_size(self, model) -> str:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        if total_size_mb > 1024:
            return f"{total_size_mb/1024:.2f} GB"
        else:
            return f"{total_size_mb:.2f} MB"
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "model_size": self._get_model_size(model),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype)
        }
