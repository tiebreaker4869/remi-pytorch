import torch
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import pickle
import utils
import time
from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
import os
import argparse
import json
import random
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

def parse_opt():
    parser = argparse.ArgumentParser()
    # Model opts
    parser.add_argument('--order', type=int,
                        help='Markov chain order (n-gram size).', default=3)
    parser.add_argument('--smoothing', type=str, choices=['none', 'laplace', 'kneser_ney'],
                        help='Smoothing method.', default='laplace')
    parser.add_argument('--alpha', type=float,
                        help='Smoothing parameter.', default=0.01)
    
    # Data opts
    parser.add_argument('--dict_path', type=str,
                        help='Dictionary path.', default='./dictionary/dictionary_REMI-tempo-checkpoint.pkl')
    parser.add_argument('--model_save_path', type=str,
                        help='Path to save Markov model.', default='./markov_models/markov_model.pkl')
    
    # Training/Testing
    parser.add_argument('--is_train', type=int,
                        help='1 for training, 0 for testing.', default=1)
    
    # Generation opts
    parser.add_argument('--prompt', type=int,
                        help='0 for generating from scratch, 1 for continue generating.', default=0)
    parser.add_argument('--prompt_path', type=str,
                        help='Prompt MIDI file path.', default='')
    parser.add_argument('--n_target_bar', type=int,
                        help='Target number of bars to generate.', default=16)
    parser.add_argument('--temperature', type=float,
                        help='Temperature for sampling (higher = more random).', default=1.0)
    parser.add_argument('--output_path', type=str,
                        help='Output path for generated MIDI.', default='./results/markov_generated.midi')
    
    # Analysis opts
    parser.add_argument('--analyze', type=int,
                        help='1 to perform statistical analysis.', default=0)
    parser.add_argument('--top_k_analysis', type=int,
                        help='Show top K n-grams in analysis.', default=20)
    
    args = parser.parse_args()
    return args

class REMIMarkovChain:
    """N-order Markov Chain for REMI token generation"""
    
    def __init__(self, order=3, smoothing='laplace', alpha=0.01):
        self.order = order
        self.smoothing = smoothing
        self.alpha = alpha
        
        # N-gram storage: (n-1)-gram -> {next_token: count}
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)  # Total counts for each context
        
        # Vocabulary
        self.vocab = set()
        self.vocab_size = 0
        
        # Statistics
        self.total_tokens = 0
        self.total_sequences = 0
        
    def train(self, sequences: List[List[int]]):
        """Train the Markov chain on sequences of tokens"""
        print(f"Training {self.order}-order Markov chain on {len(sequences)} sequences...")
        
        self.total_sequences = len(sequences)
        
        for sequence in tqdm(sequences, desc="Processing sequences"):
            self.total_tokens += len(sequence)
            
            # Add tokens to vocabulary
            self.vocab.update(sequence)
            
            # Extract n-grams
            for i in range(len(sequence) - self.order + 1):
                # Context: (order-1) tokens
                context = tuple(sequence[i:i + self.order - 1])
                # Next token
                next_token = sequence[i + self.order - 1]
                
                # Update counts
                self.ngram_counts[context][next_token] += 1
                self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total tokens processed: {self.total_tokens}")
        print(f"Unique {self.order-1}-grams: {len(self.ngram_counts)}")
        
    def get_probability(self, context: Tuple[int, ...], next_token: int) -> float:
        """Get probability of next_token given context with smoothing"""
        
        if self.smoothing == 'none':
            # No smoothing - simple MLE
            if context not in self.ngram_counts:
                return 0.0
            total_count = self.context_counts[context]
            if total_count == 0:
                return 0.0
            return self.ngram_counts[context][next_token] / total_count
        
        elif self.smoothing == 'laplace':
            # Laplace (add-alpha) smoothing
            numerator = self.ngram_counts[context][next_token] + self.alpha
            denominator = self.context_counts[context] + self.alpha * self.vocab_size
            return numerator / denominator if denominator > 0 else 1.0 / self.vocab_size
        
        elif self.smoothing == 'kneser_ney':
            # Simplified Kneser-Ney smoothing
            discount = 0.75
            context_count = self.context_counts[context]
            token_count = self.ngram_counts[context][next_token]
            
            if context_count == 0:
                return 1.0 / self.vocab_size
            
            # Main term
            prob = max(token_count - discount, 0) / context_count
            
            # Backoff term (simplified)
            lambda_val = discount * len(self.ngram_counts[context]) / context_count
            backoff_prob = 1.0 / self.vocab_size  # Uniform fallback
            
            return prob + lambda_val * backoff_prob
        
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing}")
    
    def get_next_token_distribution(self, context: Tuple[int, ...]) -> Dict[int, float]:
        """Get probability distribution for next token given context"""
        distribution = {}
        
        # Get all possible next tokens
        if context in self.ngram_counts:
            candidates = set(self.ngram_counts[context].keys())
        else:
            candidates = set()
        
        # Add all vocabulary for smoothing
        if self.smoothing != 'none':
            candidates.update(self.vocab)
        
        # Calculate probabilities
        for token in candidates:
            distribution[token] = self.get_probability(context, token)
        
        return distribution
    
    def sample_next_token(self, context: Tuple[int, ...], temperature: float = 1.0) -> int:
        """Sample next token given context with temperature"""
        distribution = self.get_next_token_distribution(context)
        
        if not distribution:
            # Fallback to random token from vocabulary
            return random.choice(list(self.vocab))
        
        # Apply temperature
        if temperature != 1.0:
            for token in distribution:
                distribution[token] = distribution[token] ** (1.0 / temperature)
        
        # Normalize
        total_prob = sum(distribution.values())
        if total_prob == 0:
            return random.choice(list(distribution.keys()))
        
        for token in distribution:
            distribution[token] /= total_prob
        
        # Sample
        tokens = list(distribution.keys())
        probs = list(distribution.values())
        
        return np.random.choice(tokens, p=probs)
    
    def generate_sequence(self, max_length: int = 1000, 
                         seed_context: Optional[List[int]] = None,
                         temperature: float = 1.0,
                         stop_tokens: Optional[List[int]] = None) -> List[int]:
        """Generate a sequence using the Markov chain"""
        
        if seed_context is None:
            # Start with most common context
            if not self.context_counts:
                return []
            most_common_context = max(self.context_counts.keys(), 
                                    key=lambda x: self.context_counts[x])
            sequence = list(most_common_context)
        else:
            sequence = seed_context.copy()
            # Ensure we have enough context
            while len(sequence) < self.order - 1:
                if self.vocab:
                    sequence.insert(0, random.choice(list(self.vocab)))
                else:
                    break
        
        if stop_tokens is None:
            stop_tokens = []
        
        # Generate tokens
        for _ in range(max_length - len(sequence)):
            if len(sequence) < self.order - 1:
                break
                
            # Get context
            context = tuple(sequence[-(self.order - 1):])
            
            # Sample next token
            next_token = self.sample_next_token(context, temperature)
            sequence.append(next_token)
            
            # Check for stop condition
            if next_token in stop_tokens:
                break
        
        return sequence
    
    def calculate_perplexity(self, test_sequences: List[List[int]]) -> float:
        """Calculate perplexity on test sequences"""
        total_log_prob = 0.0
        total_tokens = 0
        
        for sequence in test_sequences:
            for i in range(len(sequence) - self.order + 1):
                context = tuple(sequence[i:i + self.order - 1])
                next_token = sequence[i + self.order - 1]
                
                prob = self.get_probability(context, next_token)
                if prob > 0:
                    total_log_prob += np.log(prob)
                else:
                    total_log_prob += np.log(1e-10)  # Small epsilon for zero prob
                total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = np.exp(-avg_log_prob)
        return perplexity
    
    def analyze_model(self, top_k: int = 20):
        """Analyze the trained Markov model"""
        print("\n" + "="*80)
        print("MARKOV CHAIN ANALYSIS")
        print("="*80)
        
        print(f"Model Order: {self.order}")
        print(f"Smoothing: {self.smoothing} (alpha={self.alpha})")
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Total Training Tokens: {self.total_tokens}")
        print(f"Total Training Sequences: {self.total_sequences}")
        print(f"Unique {self.order-1}-grams: {len(self.ngram_counts)}")
        
        # Most common contexts
        print(f"\nTop {top_k} Most Frequent {self.order-1}-grams:")
        sorted_contexts = sorted(self.context_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:top_k]
        
        for i, (context, count) in enumerate(sorted_contexts, 1):
            print(f"{i:2d}. {context} -> {count} occurrences")
        
        # Analysis by context length
        context_lengths = defaultdict(int)
        for context in self.ngram_counts:
            context_lengths[len(context)] += 1
        
        print(f"\nContext Length Distribution:")
        for length, count in sorted(context_lengths.items()):
            print(f"Length {length}: {count} contexts")
        
        # Transition analysis for most common contexts
        print(f"\nTransition Analysis for Top 5 Contexts:")
        for context, count in sorted_contexts[:5]:
            transitions = self.ngram_counts[context].most_common(5)
            print(f"\nContext {context} (total: {count}):")
            for token, token_count in transitions:
                prob = token_count / count
                print(f"  -> {token}: {token_count} ({prob:.3f})")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'order': self.order,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocab': list(self.vocab),
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'total_sequences': self.total_sequences
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.order = model_data['order']
        self.smoothing = model_data['smoothing']
        self.alpha = model_data['alpha']
        self.ngram_counts = defaultdict(Counter, model_data['ngram_counts'])
        self.context_counts = defaultdict(int, model_data['context_counts'])
        self.vocab = set(model_data['vocab'])
        self.vocab_size = model_data['vocab_size']
        self.total_tokens = model_data['total_tokens']
        self.total_sequences = model_data['total_sequences']
        
        print(f"Model loaded from: {filepath}")
        print(f"Order: {self.order}, Vocab size: {self.vocab_size}")

class REMIMarkovDataProcessor:
    """Process REMI data for Markov chain training"""
    
    def __init__(self, dict_path: str):
        self.dict_path = dict_path
        self.event2word, self.word2event = pickle.load(open(dict_path, 'rb'))
        
    def extract_events(self, input_path: str):
        """Extract events from MIDI file"""
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        
        if 'chord' in self.dict_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
            
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
    
    def process_midi_files(self, midi_paths: List[str]) -> List[List[int]]:
        """Process MIDI files to token sequences"""
        print(f'Processing {len(midi_paths)} MIDI files...')
        
        all_sequences = []
        
        for path in tqdm(midi_paths, desc="Processing MIDI files"):
            try:
                events = self.extract_events(path)
                
                # Convert events to words
                words = []
                for event in events:
                    e = f'{event.name}_{event.value}'
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        # Handle OOV
                        if event.name == 'Note Velocity':
                            words.append(self.event2word['Note Velocity_21'])
                        else:
                            print(f'OOV token: {e}')
                
                if words:  # Only add non-empty sequences
                    all_sequences.append(words)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        print(f'Successfully processed {len(all_sequences)} sequences')
        return all_sequences

def train_markov_model():
    """Train the Markov chain model"""
    print("Starting Markov Chain Training...")
    
    # Load training data
    train_files = glob.glob('./data/POP909/*.mid')
    print(f'Found {len(train_files)} training files')
    
    if len(train_files) == 0:
        print("No training files found! Please check the data path.")
        return
    
    # Process data
    processor = REMIMarkovDataProcessor(opt.dict_path)
    sequences = processor.process_midi_files(train_files)
    
    if len(sequences) == 0:
        print("No sequences processed! Please check data processing.")
        return
    
    # Split data for validation
    split_idx = int(len(sequences) * 0.9)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Train model
    model = REMIMarkovChain(
        order=opt.order,
        smoothing=opt.smoothing,
        alpha=opt.alpha
    )
    
    model.train(train_sequences)
    
    # Calculate validation perplexity
    if val_sequences:
        val_perplexity = model.calculate_perplexity(val_sequences)
        print(f"\nValidation Perplexity: {val_perplexity:.2f}")
    
    # Save model
    model.save_model(opt.model_save_path)
    
    # Analysis
    if opt.analyze:
        model.analyze_model(opt.top_k_analysis)
    
    return model

def generate_music():
    """Generate music using trained Markov model"""
    print("Generating music with Markov Chain...")
    
    # Load model
    model = REMIMarkovChain()
    model.load_model(opt.model_save_path)
    
    # Load dictionary
    processor = REMIMarkovDataProcessor(opt.dict_path)
    
    # Prepare seed context
    seed_context = None
    if opt.prompt and opt.prompt_path and os.path.exists(opt.prompt_path):
        print(f"Using prompt from: {opt.prompt_path}")
        events = processor.extract_events(opt.prompt_path)
        seed_tokens = []
        for event in events:
            e = f'{event.name}_{event.value}'
            if e in processor.event2word:
                seed_tokens.append(processor.event2word[e])
        
        if seed_tokens:
            # Use last (order-1) tokens as seed context
            seed_context = seed_tokens[-(opt.order-1):]
        
    # Generate sequence
    bar_token = processor.event2word.get('Bar_None', None)
    stop_tokens = [bar_token] if bar_token else []
    
    max_length = 2000  # Maximum sequence length
    
    generated_tokens = model.generate_sequence(
        max_length=max_length,
        seed_context=seed_context,
        temperature=opt.temperature,
        stop_tokens=stop_tokens if opt.n_target_bar > 0 else []
    )
    
    print(f"Generated {len(generated_tokens)} tokens")
    
    # Count bars and truncate if needed
    if bar_token and opt.n_target_bar > 0:
        bar_count = 0
        truncate_idx = len(generated_tokens)
        
        for i, token in enumerate(generated_tokens):
            if token == bar_token:
                bar_count += 1
                if bar_count >= opt.n_target_bar:
                    truncate_idx = i + 1
                    break
        
        generated_tokens = generated_tokens[:truncate_idx]
        print(f"Truncated to {len(generated_tokens)} tokens ({bar_count} bars)")
    
    # Create output directory
    os.makedirs(os.path.dirname(opt.output_path) if os.path.dirname(opt.output_path) else './results', exist_ok=True)
    
    # Write MIDI
    if opt.prompt and seed_context:
        # Only save newly generated part
        new_tokens = generated_tokens[len(seed_context):]
        utils.write_midi(
            words=new_tokens,
            word2event=processor.word2event,
            output_path=opt.output_path,
            prompt_path=opt.prompt_path
        )
    else:
        utils.write_midi(
            words=generated_tokens,
            word2event=processor.word2event,
            output_path=opt.output_path,
            prompt_path=None
        )
    
    print(f"Generated music saved to: {opt.output_path}")

def compare_models():
    """Compare different Markov model configurations"""
    print("Comparing different Markov model configurations...")
    
    # Load data
    train_files = glob.glob('./data/train/*.midi')[:100]  # Limit for quick comparison
    processor = REMIMarkovDataProcessor(opt.dict_path)
    sequences = processor.process_midi_files(train_files)
    
    # Split data
    split_idx = int(len(sequences) * 0.8)
    train_seqs = sequences[:split_idx]
    test_seqs = sequences[split_idx:]
    
    # Test configurations
    configs = [
        {'order': 2, 'smoothing': 'none'},
        {'order': 2, 'smoothing': 'laplace', 'alpha': 0.01},
        {'order': 3, 'smoothing': 'none'},
        {'order': 3, 'smoothing': 'laplace', 'alpha': 0.01},
        {'order': 4, 'smoothing': 'laplace', 'alpha': 0.01},
        {'order': 3, 'smoothing': 'kneser_ney', 'alpha': 0.01},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config}")
        
        model = REMIMarkovChain(**config)
        model.train(train_seqs)
        
        perplexity = model.calculate_perplexity(test_seqs)
        
        results.append({
            'config': config,
            'perplexity': perplexity,
            'vocab_size': model.vocab_size,
            'unique_contexts': len(model.ngram_counts)
        })
        
        print(f"Perplexity: {perplexity:.2f}")
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    results.sort(key=lambda x: x['perplexity'])
    
    for i, result in enumerate(results, 1):
        config = result['config']
        print(f"{i}. Order: {config['order']}, "
              f"Smoothing: {config['smoothing']}, "
              f"Perplexity: {result['perplexity']:.2f}")

def main(opt):
    if opt.is_train:
        train_markov_model()
    else:
        generate_music()

if __name__ == '__main__':
    opt = parse_opt()
    
    # Special modes
    if opt.analyze and not opt.is_train:
        # Load and analyze existing model
        model = REMIMarkovChain()
        model.load_model(opt.model_save_path)
        model.analyze_model(opt.top_k_analysis)
    elif hasattr(opt, 'compare') and opt.compare:
        compare_models()
    else:
        main(opt)