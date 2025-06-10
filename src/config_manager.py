"""
Configuration management for PEFT experiments.
"""

import yaml
import json
from typing import Dict, Any
from pathlib import Path


class ConfigManager:
    """Manages configuration for PEFT experiments."""
    
    def __init__(self):
        self.default_config = {
            # Model settings
            "model_name": "distilbert-base-uncased",
            "num_labels": 2,
            "max_length": 128,
            
            # Dataset settings
            "train_sample_size": 5000,
            "validation_sample_size": 500,
            
            # Training settings
            "epochs": 3,
            "batch_size": 16,
            "eval_batch_size": 32,
            "learning_rate": 5e-4,
            "weight_decay": 0.01,
            "save_steps": 500,
            
            # LoRA settings
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"],
            "lora_bias": "none",
            
            # QLoRA settings
            "qlora_batch_size": 8,
            "qlora_learning_rate": 1e-4,
            "qlora_r": 8,
            "qlora_alpha": 16,
            "qlora_dropout": 0.05,
            "qlora_target_modules": ["q_lin", "v_lin"],
            
            # Evaluation settings
            "metrics": ["accuracy", "f1", "precision", "recall"],
            
            # Output settings
            "output_dir": "./peft_output",
            "save_visualizations": True,
            "save_models": True,
            
            # Experiment settings
            "seed": 42,
            "use_fp16": True,
            "gradient_accumulation_steps": 1
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration."""
        return self.default_config.copy()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Merge with default config
        merged_config = self.default_config.copy()
        merged_config.update(config)
        
        return merged_config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            # Default to YAML
            config_path = config_path.with_suffix('.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
        """
        validated_config = config.copy()
        
        # Validate numeric ranges
        if validated_config.get('lora_r', 0) <= 0:
            raise ValueError("lora_r must be positive")
        
        if validated_config.get('lora_alpha', 0) <= 0:
            raise ValueError("lora_alpha must be positive")
        
        if not 0 <= validated_config.get('lora_dropout', 0) <= 1:
            raise ValueError("lora_dropout must be between 0 and 1")
        
        if validated_config.get('epochs', 0) <= 0:
            raise ValueError("epochs must be positive")
        
        if validated_config.get('batch_size', 0) <= 0:
            raise ValueError("batch_size must be positive")
        
        if validated_config.get('learning_rate', 0) <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Validate model name
        valid_models = [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2"
        ]
        
        if validated_config.get('model_name') not in valid_models:
            print(f"Warning: {validated_config.get('model_name')} not in tested models: {valid_models}")
        
        # Validate target modules
        if not isinstance(validated_config.get('lora_target_modules'), list):
            raise ValueError("lora_target_modules must be a list")
        
        return validated_config
    
    def get_lora_config_variants(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined LoRA configuration variants for comparison.
        
        Returns:
            Dictionary of LoRA configuration variants
        """
        return {
            "low_rank": {
                "name": "Low Rank (r=4)",
                "params": {
                    "r": 4,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"]
                }
            },
            "standard": {
                "name": "Standard (r=16)",
                "params": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"]
                }
            },
            "high_rank": {
                "name": "High Rank (r=32)",
                "params": {
                    "r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"]
                }
            },
            "query_only": {
                "name": "Query-Only",
                "params": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_lin"]
                }
            },
            "with_bias": {
                "name": "With Bias Training",
                "params": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"],
                    "bias": "lora_only"
                }
            },
            "high_dropout": {
                "name": "High Dropout",
                "params": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.3,
                    "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"]
                }
            }
        }
    
    def create_experiment_config(
        self,
        base_config: Dict[str, Any],
        experiment_name: str,
        overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create configuration for a specific experiment.
        
        Args:
            base_config: Base configuration
            experiment_name: Name of the experiment
            overrides: Configuration overrides
            
        Returns:
            Experiment configuration
        """
        exp_config = base_config.copy()
        
        # Add experiment metadata
        exp_config['experiment_name'] = experiment_name
        exp_config['output_dir'] = f"{base_config['output_dir']}/{experiment_name}"
        
        # Apply overrides
        if overrides:
            exp_config.update(overrides)
        
        return self.validate_config(exp_config)
    
    def export_config_template(self, output_path: str):
        """
        Export a configuration template with comments.
        
        Args:
            output_path: Path to save the template
        """
        template_content = """# PEFT Experiment Configuration Template

# Model Configuration
model_name: "distilbert-base-uncased"  # Base model from HuggingFace
num_labels: 2  # Number of classification labels
max_length: 128  # Maximum sequence length

# Dataset Configuration
train_sample_size: 5000  # Number of training examples to use
validation_sample_size: 500  # Number of validation examples for quick eval

# Training Configuration
epochs: 3  # Number of training epochs
batch_size: 16  # Training batch size
eval_batch_size: 32  # Evaluation batch size
learning_rate: 0.0005  # Learning rate
weight_decay: 0.01  # Weight decay for regularization
save_steps: 500  # Save checkpoint every N steps

# LoRA Configuration
lora_r: 16  # Rank of LoRA adaptation matrices
lora_alpha: 32  # LoRA scaling factor (typically 2 * lora_r)
lora_dropout: 0.1  # Dropout for LoRA layers
lora_target_modules:  # Modules to apply LoRA to
  - "q_lin"
  - "v_lin"
  - "k_lin"
  - "out_lin"
lora_bias: "none"  # Bias training strategy: "none", "all", or "lora_only"

# QLoRA Configuration (for quantized training)
qlora_batch_size: 8  # Smaller batch size for memory efficiency
qlora_learning_rate: 0.0001  # Lower learning rate for stability
qlora_r: 8  # Lower rank for QLoRA
qlora_alpha: 16  # Lower alpha for QLoRA
qlora_dropout: 0.05  # Lower dropout for stability
qlora_target_modules:  # Conservative target modules for QLoRA
  - "q_lin"
  - "v_lin"

# Evaluation Configuration
metrics:  # Metrics to compute during evaluation
  - "accuracy"
  - "f1"
  - "precision"
  - "recall"

# Output Configuration
output_dir: "./peft_output"  # Base output directory
save_visualizations: true  # Whether to save plots
save_models: true  # Whether to save trained models

# Experiment Configuration
seed: 42  # Random seed for reproducibility
use_fp16: true  # Use mixed precision training (if GPU available)
gradient_accumulation_steps: 1  # Gradient accumulation steps
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        print(f"Configuration template saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Create config manager
    config_manager = ConfigManager()
    
    # Get default config
    default_config = config_manager.get_default_config()
    print("Default configuration:")
    print(json.dumps(default_config, indent=2))
    
    # Save default config as template
    config_manager.save_config(default_config, "config/default_config.yaml")
    
    # Export template with comments
    config_manager.export_config_template("config/config_template.yaml")
    
    # Test validation
    try:
        validated = config_manager.validate_config(default_config)
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Get LoRA variants
    variants = config_manager.get_lora_config_variants()
    print(f"Available LoRA variants: {list(variants.keys())}")
