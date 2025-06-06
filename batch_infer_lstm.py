import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import utils
from torch import nn

class REMILSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(REMILSTMModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch, seq_len, hidden_size)
        
        # Dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Output projection
        output = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

def temperature_sampling_batch(logits, temperature, topk, device):
    logits = logits / temperature
    
    if topk > 0 and topk < logits.size(-1):
        topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        
        sampled_indices = torch.multinomial(probs, 1).squeeze(-1)  # (batch_size,)
        next_tokens = torch.gather(topk_indices, 1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    else:
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1).squeeze(-1)
    
    return next_tokens

class LSTMBatchGenerator:
    def __init__(self, model_path, dict_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.event2word, self.word2event = pickle.load(open(dict_path, 'rb'))
        self.vocab_size = len(self.event2word)
        
        self.model = self._load_model(model_path)
        
        self._precompute_tokens()
        
        print(f"Model loaded! Vocab size: {self.vocab_size}")
    
    def _load_model(self, model_path):
        print("Loading LSTM model...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = REMILSTMModel(
                vocab_size=config['vocab_size'],
                embed_size=config.get('embed_size', 256),
                hidden_size=config.get('hidden_size', 512),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.3)
            )
        else:
            model = REMILSTMModel(
                vocab_size=self.vocab_size,
                embed_size=256,
                hidden_size=512,
                num_layers=3,
                dropout=0.3
            )
        
        model.load_state_dict(checkpoint['model'])
        model = model.to(self.device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _precompute_tokens(self):
        self.bar_token = self.event2word['Bar_None']
        self.position_token = self.event2word['Position_1/16']
        
        self.tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
        self.tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
        
        self.has_chord = any('Chord' in k for k in self.event2word.keys())
        if self.has_chord:
            self.chord_tokens = [v for k, v in self.event2word.items() if 'Chord' in k]
    
    def _initialize_sequences(self, batch_size, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        sequences = []
        for i in range(batch_size):
            if seed is not None:
                np.random.seed(seed + i)
            
            words = [self.bar_token, self.position_token]
            
            if self.has_chord:
                words.append(np.random.choice(self.chord_tokens))
                words.append(self.position_token)
            
            words.append(np.random.choice(self.tempo_classes))
            words.append(np.random.choice(self.tempo_values))
            
            sequences.append(words)
        
        return sequences
    
    def generate_batch(self, batch_size=8, max_length=2048, n_target_bar=16, 
                      temperature=1.0, topk=10, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"开始生成 {batch_size} 首音乐，目标 {n_target_bar} 小节")
        
        sequences = self._initialize_sequences(batch_size, seed)
        
        generated_sequences = []
        
        for seq_idx in range(batch_size):
            current_seq = sequences[seq_idx].copy()
            bars_generated = 0
            
            hidden = self.model.init_hidden(1, self.device)
            
            with torch.no_grad():
                generation_steps = 0
                while bars_generated < n_target_bar and len(current_seq) < max_length:
                    generation_steps += 1
                    
                    input_seq = torch.tensor([current_seq], device=self.device, dtype=torch.long)
                    
                    outputs, hidden = self.model(input_seq, hidden)
                    
                    last_logits = outputs[0, -1, :]  # (vocab_size,)
                    
                    next_token = temperature_sampling_batch(
                        last_logits.unsqueeze(0), temperature, topk, self.device
                    )[0].item()
                    
                    current_seq.append(next_token)
                    
                    if next_token == self.bar_token:
                        bars_generated += 1
                        if bars_generated >= n_target_bar:
                            break
                    
                    if generation_steps > max_length * 2:
                        break
            
            original_length = len(sequences[seq_idx])
            generated_part = current_seq[original_length:]
            generated_sequences.append(generated_part)
            
        
        return generated_sequences
    
    def batch_generate_and_save(self, total_pieces, batch_size=8, max_length=2048,
                               n_target_bar=16, temperature=1.0, topk=10,
                               output_dir='./lstm_batch_results', seed=42):
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting LSTM batch generation on {self.device}")
        print(f"Target: {total_pieces} pieces, batch size: {batch_size}")
        print(f"Bars per piece: {n_target_bar}, Temperature: {temperature}, Top-k: {topk}")
        print("-" * 60)
        
        successful = 0
        failed = 0
        total_time = 0
        piece_id = 0
        
        num_batches = (total_pieces + batch_size - 1) // batch_size
        
        pbar = tqdm(range(num_batches), desc="Generating batches")
        
        for batch_idx in pbar:
            current_batch_size = min(batch_size, total_pieces - batch_idx * batch_size)
            if current_batch_size <= 0:
                break
            
            start_time = time.time()
            
            try:
                generated_sequences = self.generate_batch(
                    batch_size=current_batch_size,
                    max_length=max_length,
                    n_target_bar=n_target_bar,
                    temperature=temperature,
                    topk=topk,
                    seed=seed + batch_idx
                )
                
                for seq_idx, generated_tokens in enumerate(generated_sequences):
                    try:
                        output_path = os.path.join(output_dir, f"lstm_generated_{piece_id:06d}.midi")
                        
                        utils.write_midi(
                            words=generated_tokens,
                            word2event=self.word2event,
                            output_path=output_path,
                            prompt_path=None
                        )
                        
                        successful += 1
                        piece_id += 1
                        
                    except Exception as e:
                        print(f"\nError saving piece {piece_id}: {e}")
                        failed += 1
                        piece_id += 1
                
                batch_time = time.time() - start_time
                total_time += batch_time
                
                avg_time_per_batch = total_time / (batch_idx + 1)
                avg_time_per_piece = total_time / max(successful, 1)
                
                pbar.set_postfix({
                    'Success': successful,
                    'Failed': failed,
                    'Batch Time': f'{batch_time:.2f}s',
                    'Avg/Piece': f'{avg_time_per_piece:.2f}s'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                failed += current_batch_size
                piece_id += current_batch_size
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        print("\n" + "="*60)
        print("LSTM BATCH GENERATION COMPLETED!")
        print(f"Successful: {successful}/{total_pieces}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total_pieces*100:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per piece: {total_time/successful:.2f} seconds" if successful > 0 else "N/A")
        print(f"Average time per batch: {total_time/num_batches:.2f} seconds")
        print(f"Output directory: {output_dir}")
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM Batch Music Generation')
    
    # Model settings
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained LSTM model checkpoint')
    parser.add_argument('--dict_path', type=str, required=True,
                        help='Path to dictionary file')
    
    # Generation settings
    parser.add_argument('--num_pieces', type=int, default=500,
                        help='Number of pieces to generate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--n_target_bar', type=int, default=16,
                        help='Number of bars to generate per piece')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature')
    parser.add_argument('--topk', type=int, default=10,
                        help='Top-k sampling parameter')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./generated_midis_lstm',
                        help='Output directory for generated MIDI files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    generator = LSTMBatchGenerator(
        model_path=args.model_path,
        dict_path=args.dict_path,
        device=args.device
    )
    
    generator.batch_generate_and_save(
        total_pieces=args.num_pieces,
        batch_size=args.batch_size,
        max_length=args.max_length,
        n_target_bar=args.n_target_bar,
        temperature=args.temperature,
        topk=args.topk,
        output_dir=args.output_dir,
        seed=args.seed
    )

if __name__ == '__main__':
    main()