"""
Model evaluation and comparison utilities for PEFT experiments.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Tuple
from transformers import Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc
)
import json
import os


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_results = {}
    
    def evaluate_model(
        self,
        model,
        eval_dataset,
        model_name: str,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Model to evaluate
            eval_dataset: Evaluation dataset
            model_name: Name for storing results
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Basic metrics
        basic_metrics = self._get_basic_metrics(model, eval_dataset, batch_size)
        
        # Detailed predictions
        predictions, true_labels, probabilities = self._get_predictions(
            model, eval_dataset, batch_size
        )
        
        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(
            predictions, true_labels, probabilities
        )
        
        # Model information
        model_info = self._get_model_info(model)
        
        # Combine all metrics
        results = {
            **basic_metrics,
            **advanced_metrics,
            **model_info,
            "predictions": predictions.tolist(),
            "true_labels": true_labels.tolist(),
            "probabilities": probabilities.tolist()
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        print(f"{model_name} evaluation complete:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def compare_models(
        self,
        models_dict: Dict[str, Any],
        eval_dataset,
        baseline_name: str = "Base Model"
    ) -> pd.DataFrame:
        """
        Compare multiple models and generate comparison table.
        
        Args:
            models_dict: Dictionary mapping model names to models
            eval_dataset: Evaluation dataset
            baseline_name: Name of baseline model for comparison
            
        Returns:
            DataFrame with comparison results
        """
        print("Comparing multiple models...")
        
        # Evaluate all models
        for model_name, model in models_dict.items():
            if model_name not in self.evaluation_results:
                self.evaluate_model(model, eval_dataset, model_name)
        
        # Create comparison DataFrame
        comparison_data = []
        baseline_metrics = self.evaluation_results.get(baseline_name, {})
        
        for model_name, results in self.evaluation_results.items():
            row = {"Model": model_name}
            
            # Core metrics
            for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
                if metric in results:
                    row[metric.capitalize()] = results[metric]
                    
                    # Calculate improvement vs baseline
                    if baseline_name in self.evaluation_results and metric in baseline_metrics:
                        baseline_value = baseline_metrics[metric]
                        if baseline_value > 0:
                            improvement = (results[metric] - baseline_value) / baseline_value * 100
                            row[f"{metric.capitalize()} vs Baseline (%)"] = improvement
            
            # Parameter efficiency
            if "trainable_params" in results and "total_params" in results:
                row["Trainable Params"] = results["trainable_params"]
                row["Total Params"] = results["total_params"]
                row["Param Efficiency (%)"] = (results["trainable_params"] / results["total_params"]) * 100
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def generate_classification_report(
        self,
        model_name: str,
        class_names: List[str] = None
    ) -> str:
        """
        Generate detailed classification report for a model.
        
        Args:
            model_name: Name of the model
            class_names: Names of the classes
            
        Returns:
            Classification report string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        predictions = np.array(results["predictions"])
        true_labels = np.array(results["true_labels"])
        
        if class_names is None:
            class_names = ["Negative", "Positive"]
        
        report = classification_report(
            true_labels, predictions, target_names=class_names
        )
        
        return report
    
    def calculate_confusion_matrix(
        self,
        model_name: str
    ) -> np.ndarray:
        """
        Calculate confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confusion matrix
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        predictions = np.array(results["predictions"])
        true_labels = np.array(results["true_labels"])
        
        return confusion_matrix(true_labels, predictions)
    
    def analyze_prediction_confidence(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze prediction confidence distribution.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confidence analysis results
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        probabilities = np.array(results["probabilities"])
        predictions = np.array(results["predictions"])
        true_labels = np.array(results["true_labels"])
        
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Separate by correctness
        correct_mask = predictions == true_labels
        correct_confidences = confidence_scores[correct_mask]
        incorrect_confidences = confidence_scores[~correct_mask]
        
        analysis = {
            "overall_mean_confidence": float(np.mean(confidence_scores)),
            "overall_std_confidence": float(np.std(confidence_scores)),
            "correct_mean_confidence": float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0,
            "incorrect_mean_confidence": float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0,
            "high_confidence_threshold": 0.9,
            "high_confidence_accuracy": float(np.mean(predictions[confidence_scores > 0.9] == true_labels[confidence_scores > 0.9])) if np.sum(confidence_scores > 0.9) > 0 else 0,
            "low_confidence_threshold": 0.6,
            "low_confidence_count": int(np.sum(confidence_scores < 0.6))
        }
        
        return analysis
    
    def calculate_parameter_efficiency(
        self,
        models_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate parameter efficiency metrics for multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to models
            
        Returns:
            DataFrame with parameter efficiency metrics
        """
        efficiency_data = []
        
        for model_name, model in models_dict.items():
            info = self._get_model_info(model)
            
            row = {
                "Model": model_name,
                "Total Parameters": info["total_params"],
                "Trainable Parameters": info["trainable_params"],
                "Trainable %": info["trainable_percentage"],
                "Memory (MB)": info.get("memory_mb", 0),
                "Reduction Factor": info["total_params"] / info["trainable_params"] if info["trainable_params"] > 0 else 1
            }
            
            efficiency_data.append(row)
        
        return pd.DataFrame(efficiency_data)
    
    def export_results(
        self,
        output_dir: str,
        include_predictions: bool = False
    ):
        """
        Export evaluation results to files.
        
        Args:
            output_dir: Directory to save results
            include_predictions: Whether to include detailed predictions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary metrics
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            summary = {
                "model": model_name,
                "accuracy": results.get("accuracy", 0),
                "f1": results.get("f1", 0),
                "precision": results.get("precision", 0),
                "recall": results.get("recall", 0),
                "roc_auc": results.get("roc_auc", 0),
                "total_params": results.get("total_params", 0),
                "trainable_params": results.get("trainable_params", 0)
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/evaluation_summary.csv", index=False)
        
        # Export detailed results (without predictions if not requested)
        detailed_results = {}
        for model_name, results in self.evaluation_results.items():
            if include_predictions:
                detailed_results[model_name] = results
            else:
                # Exclude large arrays
                detailed_results[model_name] = {
                    k: v for k, v in results.items()
                    if k not in ["predictions", "true_labels", "probabilities"]
                }
        
        with open(f"{output_dir}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Results exported to {output_dir}")
    
    def _get_basic_metrics(
        self,
        model,
        eval_dataset,
        batch_size: int
    ) -> Dict[str, float]:
        """Get basic evaluation metrics using Trainer."""
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
        
        results = trainer.evaluate()
        
        # Remove eval_ prefix and runtime metrics
        cleaned_results = {}
        for key, value in results.items():
            if key.startswith("eval_") and not key.startswith("eval_runtime") and not key.startswith("eval_samples"):
                cleaned_results[key[5:]] = value  # Remove "eval_" prefix
        
        return cleaned_results
    
    def _get_predictions(
        self,
        model,
        eval_dataset,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and probabilities."""
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
        )
        
        outputs = trainer.predict(eval_dataset)
        
        # Extract predictions and probabilities
        logits = outputs.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        predictions = np.argmax(logits, axis=1)
        true_labels = outputs.label_ids
        
        return predictions, true_labels, probabilities
    
    def _calculate_advanced_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate advanced metrics like ROC AUC."""
        metrics = {}
        
        # ROC AUC (for binary classification)
        if probabilities.shape[1] == 2:
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            metrics["roc_auc"] = auc(fpr, tpr)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            metrics[f"precision_class_{i}"] = p
            metrics[f"recall_class_{i}"] = r
            metrics[f"f1_class_{i}"] = f
        
        return metrics
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get model parameter and memory information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        memory_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
            "memory_mb": memory_mb
        }
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics for Trainer."""
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


class BenchmarkRunner:
    """Run comprehensive benchmarks comparing different PEFT methods."""
    
    def __init__(self, model_loader, evaluator):
        self.model_loader = model_loader
        self.evaluator = evaluator
        self.benchmark_results = {}
    
    def run_full_benchmark(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./benchmark_results"
    ) -> Dict[str, Any]:
        """
        Run a full benchmark comparing base model, LoRA, and QLoRA.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save results
            
        Returns:
            Complete benchmark results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting comprehensive PEFT benchmark...")
        
        # Load base model
        base_model, tokenizer = self.model_loader.load_model()
        
        # Evaluate base model
        base_results = self.evaluator.evaluate_model(
            base_model, eval_dataset, "Base Model"
        )
        
        # Train and evaluate LoRA variants
        from .lora_trainer import LoRATrainer
        lora_trainer = LoRATrainer(base_model, tokenizer)
        
        lora_configs = lora_trainer.get_default_configurations()
        lora_results = lora_trainer.compare_configurations(
            train_dataset, eval_dataset, lora_configs,
            output_base_dir=f"{output_dir}/lora_variants"
        )
        
        # Try QLoRA if available
        try:
            from .qlora_trainer import QLoRATrainer
            qlora_trainer = QLoRATrainer(self.model_loader.model_name, tokenizer)
            
            if qlora_trainer.check_qlora_requirements():
                qlora_model, _ = qlora_trainer.train(
                    train_dataset, eval_dataset,
                    output_dir=f"{output_dir}/qlora_model"
                )
                qlora_results = self.evaluator.evaluate_model(
                    qlora_model, eval_dataset, "QLoRA Model"
                )
            else:
                print("QLoRA requirements not met, skipping QLoRA benchmark")
                qlora_results = None
        except Exception as e:
            print(f"QLoRA benchmark failed: {e}")
            qlora_results = None
        
        # Compile results
        self.benchmark_results = {
            "base_model": base_results,
            "lora_variants": lora_results,
            "qlora_model": qlora_results
        }
        
        # Export results
        self.evaluator.export_results(output_dir)
        
        # Save benchmark summary
        with open(f"{output_dir}/benchmark_summary.json", "w") as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        print(f"Benchmark complete! Results saved to {output_dir}")
        
        return self.benchmark_results
