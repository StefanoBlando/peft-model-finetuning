"""
Visualization utilities for PEFT experiments and results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix
from math import pi
import warnings
warnings.filterwarnings("ignore")


class PEFTVisualizer:
    """Comprehensive visualization suite for PEFT experiments."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple = (12, 8)):
        """
        Initialize the visualizer with style settings.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 12
        self.colors = sns.color_palette("viridis", 10)
        
    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict],
        metrics: List[str] = None,
        output_dir: str = None,
        baseline_name: str = "Base Model"
    ):
        """
        Create comprehensive model comparison visualizations.
        
        Args:
            results_dict: Dictionary mapping model names to results
            metrics: List of metrics to compare
            output_dir: Directory to save plots
            baseline_name: Name of baseline model
        """
        if metrics is None:
            metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            if all(metric in results.get('eval_' + metric, results.get(metric, {})) 
                  for results in results_dict.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            print("No common metrics found for comparison")
            return
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        models = list(results_dict.keys())
        baseline_idx = models.index(baseline_name) if baseline_name in models else 0
        
        for idx, metric in enumerate(available_metrics[:4]):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract values
            values = []
            for model in models:
                result = results_dict[model]
                value = result.get(f'eval_{metric}', result.get(metric, 0))
                values.append(value)
            
            # Create bar plot
            colors = ['#1f77b4' if i == baseline_idx else self.colors[i % len(self.colors)] 
                     for i in range(len(models))]
            bars = ax.bar(models, values, color=colors)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Add improvement percentages
            if baseline_name in models:
                baseline_value = values[baseline_idx]
                for i, (model, value) in enumerate(zip(models, values)):
                    if model != baseline_name and baseline_value > 0:
                        improvement = (value - baseline_value) / baseline_value * 100
                        color = 'green' if improvement > 0 else 'red'
                        ax.text(i, value * 0.95, f'{improvement:+.1f}%',
                               ha='center', va='center', color=color, fontweight='bold')
            
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {output_dir}/model_comparison.png")
        
        plt.show()
    
    def plot_confusion_matrices(
        self,
        predictions_dict: Dict[str, Dict],
        class_names: List[str] = None,
        output_dir: str = None
    ):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            predictions_dict: Dict with model predictions and true labels
            class_names: Names of the classes
            output_dir: Directory to save plots
        """
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        n_models = len(predictions_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, data) in enumerate(predictions_dict.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get predictions and true labels
            predictions = np.array(data['predictions'])
            true_labels = np.array(data['true_labels'])
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar=True)
            
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12)
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")
        
        plt.show()
    
    def plot_parameter_efficiency(
        self,
        efficiency_data: Dict[str, Dict],
        output_dir: str = None
    ):
        """
        Visualize parameter efficiency for different models.
        
        Args:
            efficiency_data: Dict with parameter information for each model
            output_dir: Directory to save plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie charts for parameter distribution
        models = list(efficiency_data.keys())
        peft_models = [m for m in models if m != "Base Model"]
        
        if len(peft_models) > 0:
            # Show parameter efficiency for first PEFT model
            model_data = efficiency_data[peft_models[0]]
            
            trainable_params = model_data.get('trainable_params', 0)
            total_params = model_data.get('total_params', 1)
            frozen_params = total_params - trainable_params
            
            # Pie chart
            sizes = [trainable_params, frozen_params]
            labels = ['Trainable Parameters', 'Frozen Parameters']
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.set_title(f'Parameter Efficiency - {peft_models[0]}', fontsize=14)
        
        # Bar chart of parameter efficiency vs performance
        model_names = []
        param_efficiencies = []
        accuracies = []
        
        for model_name, data in efficiency_data.items():
            if model_name == "Base Model":
                continue
            
            model_names.append(model_name)
            
            trainable = data.get('trainable_params', 0)
            total = data.get('total_params', 1)
            param_efficiencies.append((trainable / total) * 100)
            
            accuracy = data.get('accuracy', data.get('eval_accuracy', 0))
            accuracies.append(accuracy)
        
        if param_efficiencies and accuracies:
            scatter = ax2.scatter(param_efficiencies, accuracies, 
                                s=100, c=range(len(model_names)), cmap='viridis')
            
            # Add labels
            for i, model in enumerate(model_names):
                ax2.annotate(model, (param_efficiencies[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Add baseline reference if available
            if "Base Model" in efficiency_data:
                base_accuracy = efficiency_data["Base Model"].get('accuracy', 
                              efficiency_data["Base Model"].get('eval_accuracy', 0))
                ax2.axhline(y=base_accuracy, color='red', linestyle='--', alpha=0.7,
                           label=f'Base Model: {base_accuracy:.4f}')
                ax2.legend()
            
            ax2.set_xlabel('Percentage of Parameters Trained (%)', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Accuracy vs Parameter Efficiency', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/parameter_efficiency.png", dpi=300, bbox_inches='tight')
            print(f"Parameter efficiency plot saved to {output_dir}/parameter_efficiency.png")
        
        plt.show()
    
    def plot_training_curves(
        self,
        training_history: List[Dict],
        model_name: str = "Model",
        output_dir: str = None
    ):
        """
        Plot training loss and validation metrics over time.
        
        Args:
            training_history: List of training log entries
            model_name: Name of the model for title
            output_dir: Directory to save plots
        """
        # Extract training data
        train_loss = []
        train_steps = []
        eval_loss = []
        eval_accuracy = []
        eval_steps = []
        
        for entry in training_history:
            if 'loss' in entry and 'eval_loss' not in entry:
                train_loss.append(entry['loss'])
                train_steps.append(entry['step'])
            if 'eval_loss' in entry:
                eval_loss.append(entry['eval_loss'])
                eval_steps.append(entry['step'])
            if 'eval_accuracy' in entry:
                eval_accuracy.append(entry['eval_accuracy'])
        
        if not train_loss and not eval_accuracy:
            print("No training history data found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training loss
        if train_loss:
            axes[0].plot(train_steps, train_loss, 'b-', marker='o', 
                        markersize=4, alpha=0.7, label='Training Loss')
            axes[0].set_title(f'Training Loss - {model_name}', fontsize=14)
            axes[0].set_xlabel('Steps', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Validation metrics
        if eval_accuracy:
            axes[1].plot(eval_steps, eval_accuracy, 'g-', marker='o', 
                        markersize=6, label='Accuracy')
            
            # Add validation loss on secondary axis if available
            if eval_loss and len(eval_loss) == len(eval_accuracy):
                ax2 = axes[1].twinx()
                ax2.plot(eval_steps, eval_loss, 'r--', marker='x', 
                        markersize=4, alpha=0.7, label='Val Loss')
                ax2.set_ylabel('Loss', color='red', fontsize=12)
                ax2.tick_params(axis='y', colors='red')
            
            axes[1].set_title(f'Validation Metrics - {model_name}', fontsize=14)
            axes[1].set_xlabel('Steps', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/training_curves_{model_name.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {output_dir}/training_curves_{model_name.replace(' ', '_')}.png")
        
        plt.show()
    
    def plot_lora_configuration_comparison(
        self,
        config_results: List[Dict],
        output_dir: str = None
    ):
        """
        Compare different LoRA configurations.
        
        Args:
            config_results: List of configuration results
            output_dir: Directory to save plots
        """
        if not config_results:
            print("No configuration results to plot")
            return
        
        df = pd.DataFrame(config_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'f1', 'param_efficiency', 'training_time_min']
        titles = ['Accuracy', 'F1 Score', 'Parameter Efficiency (%)', 'Training Time (min)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            if metric not in df.columns:
                continue
                
            ax = axes[idx]
            
            # Create bar plot
            bars = ax.bar(range(len(df)), df[metric], color=self.colors[:len(df)])
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, df[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(title.split('(')[0].strip(), fontsize=12)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['config_name'], rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/lora_config_comparison.png", dpi=300, bbox_inches='tight')
            print(f"LoRA configuration comparison saved to {output_dir}/lora_config_comparison.png")
        
        plt.show()
        
        # Create radar chart for multi-dimensional comparison
        self._create_radar_chart(df, output_dir)
    
    def plot_memory_usage(
        self,
        memory_data: Dict[str, float],
        output_dir: str = None
    ):
        """
        Plot memory usage comparison between models.
        
        Args:
            memory_data: Dict mapping model names to memory usage in MB
            output_dir: Directory to save plots
        """
        models = list(memory_data.keys())
        memory_values = list(memory_data.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.bar(models, memory_values, color=self.colors[:len(models)])
        
        # Add value labels
        for bar, value in zip(bars, memory_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{value:.1f} MB', ha='center', va='bottom', fontsize=10)
        
        # Add reduction percentages if base model exists
        if "Base Model" in memory_data:
            base_memory = memory_data["Base Model"]
            for i, (model, memory) in enumerate(zip(models, memory_values)):
                if model != "Base Model":
                    reduction = (base_memory - memory) / base_memory * 100
                    ax.text(i, memory / 2, f'{reduction:+.1f}%',
                           ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to {output_dir}/memory_usage.png")
        
        plt.show()
    
    def plot_confidence_analysis(
        self,
        confidence_data: Dict[str, Any],
        model_name: str,
        output_dir: str = None
    ):
        """
        Plot prediction confidence analysis.
        
        Args:
            confidence_data: Dictionary with confidence analysis results
            model_name: Name of the model
            output_dir: Directory to save plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confidence distribution histogram
        if 'confidence_scores' in confidence_data:
            confidence_scores = confidence_data['confidence_scores']
            ax1.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidence_scores):.3f}')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Confidence Distribution - {model_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Confidence vs Accuracy
        if 'correct_confidences' in confidence_data and 'incorrect_confidences' in confidence_data:
            correct_conf = confidence_data['correct_confidences']
            incorrect_conf = confidence_data['incorrect_confidences']
            
            ax2.hist(correct_conf, bins=15, alpha=0.7, label='Correct Predictions', 
                    color='green', edgecolor='black')
            ax2.hist(incorrect_conf, bins=15, alpha=0.7, label='Incorrect Predictions', 
                    color='red', edgecolor='black')
            
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Confidence by Correctness - {model_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/confidence_analysis_{model_name.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Confidence analysis saved to {output_dir}/confidence_analysis_{model_name.replace(' ', '_')}.png")
        
        plt.show()
    
    def create_comprehensive_dashboard(
        self,
        all_results: Dict[str, Any],
        output_dir: str
    ):
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            all_results: Complete results dictionary
            output_dir: Directory to save dashboard
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating comprehensive visualization dashboard...")
        
        # Plot all comparisons
        if 'model_results' in all_results:
            self.plot_model_comparison(all_results['model_results'], output_dir=output_dir)
        
        if 'predictions' in all_results:
            self.plot_confusion_matrices(all_results['predictions'], output_dir=output_dir)
        
        if 'parameter_efficiency' in all_results:
            self.plot_parameter_efficiency(all_results['parameter_efficiency'], output_dir=output_dir)
        
        if 'lora_configs' in all_results:
            self.plot_lora_configuration_comparison(all_results['lora_configs'], output_dir=output_dir)
        
        if 'memory_usage' in all_results:
            self.plot_memory_usage(all_results['memory_usage'], output_dir=output_dir)
        
        if 'training_history' in all_results:
            for model_name, history in all_results['training_history'].items():
                self.plot_training_curves(history, model_name, output_dir=output_dir)
        
        print(f"Dashboard complete! All visualizations saved to {output_dir}")
    
    def _create_radar_chart(
        self,
        df: pd.DataFrame,
        output_dir: str = None
    ):
        """Create radar chart for LoRA configuration comparison."""
        if len(df) == 0:
            return
        
        # Select metrics for radar chart
        metrics = ['accuracy', 'f1', 'param_efficiency']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 3:
            return
        
        # Normalize data for radar chart
        normalized_data = {}
        for metric in available_metrics:
            values = df[metric].values
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized_data[metric] = (values - min_val) / (max_val - min_val)
            else:
                normalized_data[metric] = np.ones_like(values)
        
        # Number of variables
        N = len(available_metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axis per variable and add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        
        # Draw the chart for each configuration
        for i, (_, row) in enumerate(df.iterrows()):
            values = [normalized_data[metric][i] for metric in available_metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   label=row['config_name'], color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.set_title('LoRA Configuration Comparison', fontsize=15, pad=20)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/lora_radar_chart.png", dpi=300, bbox_inches='tight')
            print(f"Radar chart saved to {output_dir}/lora_radar_chart.png")
        
        plt.show()
    
    def save_summary_table(
        self,
        results_dict: Dict[str, Dict],
        output_path: str
    ):
        """
        Save a summary table of all results.
        
        Args:
            results_dict: Dictionary of model results
            output_path: Path to save the table
        """
        # Create summary data
        summary_data = []
        
        for model_name, results in results_dict.items():
            row = {"Model": model_name}
            
            # Extract key metrics
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                value = results.get(f'eval_{metric}', results.get(metric, None))
                if value is not None:
                    row[metric.capitalize()] = f"{value:.4f}"
            
            # Parameter information
            if 'trainable_params' in results:
                row['Trainable Params'] = f"{results['trainable_params']:,}"
            if 'total_params' in results:
                row['Total Params'] = f"{results['total_params']:,}"
            if 'trainable_params' in results and 'total_params' in results:
                pct = (results['trainable_params'] / results['total_params']) * 100
                row['Efficiency (%)'] = f"{pct:.2f}%"
            
            summary_data.append(row)
        
        # Create and save DataFrame
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Summary table saved to {output_path}")
        
        return df
