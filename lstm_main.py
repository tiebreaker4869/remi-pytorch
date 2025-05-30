import torch
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import miditoolkit
# import modules
import pickle
import utils
import time
from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # training opts
    parser.add_argument('--is_train', type=int,
                        help='1 for training, 0 for testing.', default=1)
    parser.add_argument('--is_continue', type=int,
                        help='1 for continue training, 0 for training from scratch.', default=0)
    parser.add_argument('--continue_pth', type=str,
                        help='Continue training checkpoint path.', default='')
    parser.add_argument('--dict_path', type=str,
                        help='Dictionary path.', default='./dictionary/dictionary_REMI-tempo-checkpoint.pkl')
    
    # LSTM model opts
    parser.add_argument('--hidden_size', type=int,
                        help='LSTM hidden size.', default=512)
    parser.add_argument('--num_layers', type=int,
                        help='Number of LSTM layers.', default=3)
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate.', default=0.3)
    parser.add_argument('--embed_size', type=int,
                        help='Embedding size.', default=256)
    parser.add_argument('--seq_len', type=int,
                        help='Sequence length.', default=512)
    
    # validation opts
    parser.add_argument('--val_freq', type=int,
                        help='Validation frequency (every N epochs).', default=5)
    parser.add_argument('--early_stop_patience', type=int,
                        help='Early stopping patience.', default=15)
    
    # training hyperparameters
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.', default=32)
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate.', default=0.001)
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs.', default=5)
    
    # testing opts
    parser.add_argument('--prompt', type=int,
                        help='0 for generating from scratch, 1 for continue generating.', default=False)
    parser.add_argument('--prompt_path', type=str,
                        help='if prompt is True, you have to specify the continue generating midi file path.', default='')
    parser.add_argument('--n_target_bar', type=int,
                        help='Control the generate result.', default=16)
    parser.add_argument('--temperature', type=float,
                        help='Control the generate result.', default=1.0)
    parser.add_argument('--topk', type=int,
                        help='Control the generate result.', default=10)
    parser.add_argument('--output_path', type=str,
                        help='output path', default='./results/lstm_from_scratch.midi')
    parser.add_argument('--model_path', type=str,
                        help='model path', default='./checkpoints/lstm_best_model.pkl')
    args = parser.parse_args()
    return args

class REMIDataset(Dataset):
    def __init__(self, midi_l=[], dict_pth='./dictionary/dictionary_REMI-tempo-checkpoint.pkl', seq_len=512):
        self.midi_l = midi_l
        self.tokenizer = REMI()
        self.checkpoint_path = dict_pth
        self.seq_len = seq_len
        self.dictionary_path = dict_pth
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.sequences = self.prepare_data(self.midi_l)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        # For LSTM: input is sequence[:-1], target is sequence[1:]
        return {
            'input': torch.tensor(sequence[:-1], dtype=torch.long),
            'target': torch.tensor(sequence[1:], dtype=torch.long)
        }
            
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        print(f'Processing {len(midi_paths)} MIDI files...')
        # extract events
        all_events = []
        for path in tqdm(midi_paths, desc="Extracting events"):
            try:
                events = self.extract_events(path)
                all_events.append(events)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        print(f'{len(all_events)} event sequences extracted.')
        
        # event to word
        all_words = []
        for events in tqdm(all_events, desc="Converting to words"):
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV handling
                    if event.name == 'Note Velocity':
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        print('OOV token: {}'.format(e))
            all_words.append(words)
        
        print(f'{len(all_words)} word sequences created.')
        
        # Create sequences for LSTM training
        sequences = []
        for words in tqdm(all_words, desc="Creating sequences"):
            # Create overlapping sequences
            for i in range(0, len(words) - self.seq_len, self.seq_len // 2):
                if i + self.seq_len < len(words):
                    sequence = words[i:i + self.seq_len + 1]  # +1 for target
                    sequences.append(sequence)
        
        print(f'{len(sequences)} sequences created for training.')
        return sequences

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
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
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

def calculate_nll(model, dataloader, device, max_batches=None):
    """Calculate negative log-likelihood on validation/test set"""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating NLL")):
            if max_batches and i >= max_batches:
                break
            
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size, device)
            
            outputs, _ = model(inputs, hidden)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            # Calculate cross-entropy loss (negative log-likelihood)
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            
            total_nll += loss.item()
            total_tokens += targets.numel()
            batch_count += 1
    
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    
    model.train()
    return avg_nll, perplexity, total_tokens

def temperature_sampling(logits, temperature, topk):
    """Apply temperature and top-k sampling"""
    logits = logits / temperature
    
    if topk > 0:
        # Top-k sampling
        topk_logits, topk_indices = torch.topk(logits, topk)
        probs = F.softmax(topk_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, 1).item()
        next_token = topk_indices[next_token_idx].item()
    else:
        # Standard sampling
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
    
    return next_token

def generate_music(model, event2word, word2event, device, max_length=1024, 
                  temperature=1.0, topk=10, prompt_tokens=None):
    """Generate music using the trained LSTM model"""
    model.eval()
    
    if prompt_tokens is None:
        # Start with Bar token
        generated = [event2word['Bar_None']]
    else:
        generated = prompt_tokens.copy()
    
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            # Prepare input
            input_seq = torch.tensor([generated], dtype=torch.long).to(device)
            
            # Forward pass
            outputs, hidden = model(input_seq, hidden)
            
            # Get last timestep logits
            logits = outputs[0, -1, :]
            
            # Sample next token
            next_token = temperature_sampling(logits, temperature, topk)
            generated.append(next_token)
            
            # Stop conditions
            if next_token == event2word.get('EOS', -1):
                break
    
    model.train()
    return generated

def test(prompt_path='./data/evaluation/000.midi', prompt=True, n_target_bar=16,
         temperature=1.0, topk=10, output_path='', model_path=''):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else './results', exist_ok=True)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load dictionary
    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    vocab_size = len(event2word)
    
    # Initialize model
    model = REMILSTMModel(
        vocab_size=vocab_size,
        embed_size=opt.embed_size,
        hidden_size=opt.hidden_size,
        num_layers=opt.num_layers,
        dropout=opt.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Prepare prompt if needed
    prompt_tokens = None
    if prompt and os.path.exists(prompt_path):
        test_data = REMIDataset(midi_l=[prompt_path], dict_pth=opt.dict_path, seq_len=opt.seq_len)
        events = test_data.extract_events(prompt_path)
        prompt_tokens = [event2word['{}_{}'.format(e.name, e.value)] for e in events]
        prompt_tokens.append(event2word['Bar_None'])
    
    print('Start generating...')
    generated_tokens = generate_music(
        model=model,
        event2word=event2word,
        word2event=word2event,
        device=device,
        max_length=2048,
        temperature=temperature,
        topk=topk,
        prompt_tokens=prompt_tokens
    )
    
    # Write MIDI
    if prompt and prompt_tokens:
        # Only save the newly generated part
        utils.write_midi(
            words=generated_tokens[len(prompt_tokens):],
            word2event=word2event,
            output_path=output_path,
            prompt_path=prompt_path
        )
    else:
        utils.write_midi(
            words=generated_tokens,
            word2event=word2event,
            output_path=output_path,
            prompt_path=None
        )
    
    print(f'Generated music saved to: {output_path}')

def train(is_continue=False, checkpoints_path=''):
    # Load arguments
    epochs = opt.epochs
    
    # Create data list
    all_train_list = glob.glob('./data/POP909/*.mid')
    print('Total MIDI files:', len(all_train_list))
    
    # Split data
    train_list, val_list = train_test_split(all_train_list, test_size=0.1, random_state=42)
    print('Train list len =', len(train_list))
    print('Validation list len =', len(val_list))
    
    # Create datasets
    train_dataset = REMIDataset(midi_l=train_list, dict_pth=opt.dict_path, seq_len=opt.seq_len)
    val_dataset = REMIDataset(midi_l=val_list, dict_pth=opt.dict_path, seq_len=opt.seq_len)
    
    # Load vocabulary
    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    vocab_size = len(event2word)
    print(f'Vocabulary size: {vocab_size}')
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    print('Dataloaders created')
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')
    
    # Create model
    if not is_continue:
        start_epoch = 1
        model = REMILSTMModel(
            vocab_size=vocab_size,
            embed_size=opt.embed_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.num_layers,
            dropout=opt.dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize tracking variables
        best_val_nll = float('inf')
        patience_counter = 0
        train_nlls = []
        val_nlls = []
        
    else:
        # Load checkpoint
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            
            model = REMILSTMModel(
                vocab_size=vocab_size,
                embed_size=opt.embed_size,
                hidden_size=opt.hidden_size,
                num_layers=opt.num_layers,
                dropout=opt.dropout
            ).to(device)
            
            model.load_state_dict(checkpoint['model'])
            
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load tracking variables
            best_val_nll = checkpoint.get('best_val_nll', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            train_nlls = checkpoint.get('train_nlls', [])
            val_nlls = checkpoint.get('val_nlls', [])
        else:
            print(f"Checkpoint file {checkpoints_path} not found!")
            return
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start training...')
    
    # Create checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
        for batch in train_bar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            batch_size = inputs.size(0)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            outputs, _ = model(inputs, hidden)
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            # Calculate loss
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item() * targets.numel()
            epoch_tokens += targets.numel()
            
            # Update progress bar
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training NLL
        train_nll = epoch_loss / epoch_tokens
        train_nlls.append(train_nll)
        
        print(f'>>> Epoch: {epoch}, Train NLL: {train_nll:.5f}')
        
        # Validation phase
        if epoch % opt.val_freq == 0 or epoch == epochs:
            print(f"Calculating validation NLL for epoch {epoch}...")
            val_nll, val_perplexity, val_tokens = calculate_nll(model, val_dataloader, device, max_batches=50)
            val_nlls.append(val_nll)
            
            print(f'>>> Epoch: {epoch}, Validation NLL: {val_nll:.5f}, Perplexity: {val_perplexity:.2f}')
            print(f'>>> Evaluated on {val_tokens} tokens')
            
            # Learning rate scheduling
            scheduler.step(val_nll)
            
            # Early stopping check
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                patience_counter = 0
                print(f'>>> New best validation NLL: {best_val_nll:.5f}')
                
                # Save best model
                best_model_path = './checkpoints/lstm_best_model.pkl'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_nll': train_nll,
                    'val_nll': val_nll,
                    'best_val_nll': best_val_nll,
                    'patience_counter': patience_counter,
                    'train_nlls': train_nlls,
                    'val_nlls': val_nlls,
                    'model_config': {
                        'vocab_size': vocab_size,
                        'embed_size': opt.embed_size,
                        'hidden_size': opt.hidden_size,
                        'num_layers': opt.num_layers,
                        'dropout': opt.dropout
                    }
                }, best_model_path)
                print(f'>>> Best model saved to {best_model_path}')
            else:
                patience_counter += 1
                print(f'>>> No improvement. Patience: {patience_counter}/{opt.early_stop_patience}')
                
                if patience_counter >= opt.early_stop_patience:
                    print(f'>>> Early stopping triggered after {patience_counter} epochs without improvement')
                    break
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'./checkpoints/lstm_epoch_{epoch:03d}.pkl'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_nll': train_nll,
                'val_nll': val_nlls[-1] if val_nlls else float('inf'),
                'best_val_nll': best_val_nll,
                'patience_counter': patience_counter,
                'train_nlls': train_nlls,
                'val_nlls': val_nlls,
                'model_config': {
                    'vocab_size': vocab_size,
                    'embed_size': opt.embed_size,
                    'hidden_size': opt.hidden_size,
                    'num_layers': opt.num_layers,
                    'dropout': opt.dropout
                }
            }, checkpoint_path)
        
        # Save training history
        np.save('lstm_training_nlls_history.npy', np.array(train_nlls))
        np.save('lstm_validation_nlls_history.npy', np.array(val_nlls))
        
        # Print summary every 10 epochs
        if epoch % 10 == 0:
            print("\n" + "="*80)
            print(f"LSTM EPOCH {epoch} SUMMARY:")
            print(f"Train NLL: {train_nll:.5f}")
            if val_nlls:
                print(f"Best Validation NLL: {best_val_nll:.5f}")
                print(f"Current Validation NLL: {val_nlls[-1]:.5f}")
            print(f"Patience Counter: {patience_counter}/{opt.early_stop_patience}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print("="*80 + "\n")
    
    # Final summary
    print("\n" + "="*80)
    print("LSTM TRAINING COMPLETED!")
    print(f"Total epochs trained: {epoch}")
    print(f"Best validation NLL: {best_val_nll:.5f}")
    print(f"Final train NLL: {train_nlls[-1]:.5f}")
    if val_nlls:
        print(f"Final validation NLL: {val_nlls[-1]:.5f}")
    print("="*80)

def main(opt):
    if opt.is_train:
        if not opt.is_continue:
            train()
        else:
            train(is_continue=opt.is_continue, checkpoints_path=opt.continue_pth)
    else:
        if not opt.prompt:
            test(prompt=opt.prompt, n_target_bar=opt.n_target_bar, 
                temperature=opt.temperature, topk=opt.topk, 
                output_path=opt.output_path, model_path=opt.model_path)
        else:
            test(prompt_path=opt.prompt_path, n_target_bar=opt.n_target_bar, 
                temperature=opt.temperature, topk=opt.topk,
                output_path=opt.output_path, model_path=opt.model_path)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)