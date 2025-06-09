"""Lightweight Fine-Tuning Project - Parameter-Efficient Fine-Tuning with LoRA and QLoRA."""

__version__ = "0.1.0"
__author__ = "Stefano Blando"
__description__ = "Parameter-Efficient Fine-Tuning for sentiment analysis using LoRA and QLoRA"

# Import main components for easy access
from .model_loader import ModelLoader
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer
from .evaluator import ModelEvaluator, ModelComparator
from .visualizer import ResultsVisualizer
from .config_manager import ConfigManager

__all__ = [
    "ModelLoader",
    "LoRATrainer", 
    "QLoRATrainer",
    "ModelEvaluator",
    "ModelComparator",
    "ResultsVisualizer",
    "ConfigManager"
]
