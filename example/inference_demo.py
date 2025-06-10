"""
Inference demo showing how to use trained PEFT models for sentiment analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lora_trainer import LoRATrainer


class SentimentPredictor:
    """Simple sentiment prediction class using PEFT models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained PEFT model.
        
        Args:
            model_path: Path to the saved PEFT model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the PEFT model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load PEFT configuration
        peft_config = PeftConfig.from_pretrained(self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=2
        )
        
        # Load PEFT weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def predict(self, text: str, return_probabilities: bool = False):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction result (label or dict with probabilities)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process output
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Format result
        label = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[0][prediction].item()
        
        if return_probabilities:
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "Negative": probabilities[0][0].item(),
                    "Positive": probabilities[0][1].item()
                }
            }
        else:
            return label
    
    def predict_batch(self, texts: list, return_probabilities: bool = False):
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probabilities)
            results.append(result)
        
        return results


def demo_basic_usage():
    """Demonstrate basic usage of the sentiment predictor."""
    print("="*60)
    print("BASIC SENTIMENT PREDICTION DEMO")
    print("="*60)
    
    # Sample texts for testing
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This movie was terrible and boring.",
        "The service was okay, nothing special.",
        "Best experience I've ever had!",
        "I hate this so much, worst purchase ever.",
        "It's fine, does what it's supposed to do."
    ]
    
    # Try to load a trained model (update path as needed)
    model_paths = [
        "./peft_output/models/lora_standard",
        "./results/models/lora_standard",
        "./models/lora_standard"
    ]
    
    predictor = None
    for path in model_paths:
        if Path(path).exists():
            try:
                predictor = SentimentPredictor(path)
                break
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
                continue
    
    if predictor is None:
        print("❌ No trained model found. Please run the training pipeline first.")
        print("Available paths checked:", model_paths)
        return
    
    # Make predictions
    print(f"\nTesting on {len(test_texts)} examples:")
    print("-" * 60)
    
    for text in test_texts:
        result = predictor.predict(text, return_probabilities=True)
        
        print(f"Text: {text}")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: Positive={result['probabilities']['Positive']:.3f}, "
              f"Negative={result['probabilities']['Negative']:.3f}")
        print("-" * 60)


def demo_interactive_mode():
    """Interactive mode for testing custom inputs."""
    print("="*60)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*60)
    print("Enter text to analyze (type 'quit' to exit)")
    
    # Load model
    model_paths = [
        "./peft_output/models/lora_standard",
        "./results/models/lora_standard",
        "./models/lora_standard"
    ]
    
    predictor = None
    for path in model_paths:
        if Path(path).exists():
            try:
                predictor = SentimentPredictor(path)
                break
            except Exception as e:
                continue
    
    if predictor is None:
        print("❌ No trained model found. Please run the training pipeline first.")
        return
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            result = predictor.predict(text, return_probabilities=True)
            
            print(f"→ Sentiment: {result['label']}")
            print(f"→ Confidence: {result['confidence']:.3f}")
            print(f"→ Probabilities: Pos={result['probabilities']['Positive']:.3f}, "
                  f"Neg={result['probabilities']['Negative']:.3f}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_comparison():
    """Compare predictions from different model variants."""
    print("="*60)
    print("MODEL COMPARISON DEMO")
    print("="*60)
    
    # Look for different model variants
    possible_models = {
        "Standard LoRA": "./peft_output/models/lora_standard",
        "High Rank LoRA": "./peft_output/models/lora_variants/variant_2",
        "QLoRA": "./peft_output/models/qlora_model"
    }
    
    # Load available models
    predictors = {}
    for name, path in possible_models.items():
        if Path(path).exists():
            try:
                predictors[name] = SentimentPredictor(path)
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    if not predictors:
        print("❌ No models found for comparison.")
        return
    
    # Test texts
    test_texts = [
        "This is absolutely fantastic!",
        "I really dislike this product.",
        "It's okay, nothing special."
    ]
    
    print(f"\nComparing predictions on {len(test_texts)} texts:")
    print("="*60)
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 40)
        
        for model_name, predictor in predictors.items():
            result = predictor.predict(text, return_probabilities=True)
            print(f"{model_name:15}: {result['label']:8} (conf: {result['confidence']:.3f})")


def benchmark_speed():
    """Benchmark inference speed."""
    print("="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    
    # Load model
    model_paths = [
        "./peft_output/models/lora_standard",
        "./results/models/lora_standard"
    ]
    
    predictor = None
    for path in model_paths:
        if Path(path).exists():
            try:
                predictor = SentimentPredictor(path)
                break
            except Exception as e:
                continue
    
    if predictor is None:
        print("❌ No trained model found.")
        return
    
    # Generate test data
    test_texts = [
        "This is a test sentence for speed benchmarking.",
        "Another example text to measure inference performance.",
        "Speed testing with various sentence lengths and complexity."
    ] * 100  # 300 examples
    
    print(f"Benchmarking on {len(test_texts)} examples...")
    
    # Warm-up
    predictor.predict("Warm-up prediction")
    
    # Benchmark
    import time
    start_time = time.time()
    
    results = predictor.predict_batch(test_texts)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per prediction: {(total_time / len(test_texts)) * 1000:.2f} ms")
    print(f"Predictions per second: {len(test_texts) / total_time:.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PEFT Sentiment Analysis Demo")
    parser.add_argument("--mode", choices=["basic", "interactive", "compare", "benchmark"],
                       default="basic", help="Demo mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        demo_basic_usage()
    elif args.mode == "interactive":
        demo_interactive_mode()
    elif args.mode == "compare":
        demo_comparison()
    elif args.mode == "benchmark":
        benchmark_speed()
    
    print("\n" + "="*60)
    print("Demo complete! Try other modes with --mode [basic|interactive|compare|benchmark]")
