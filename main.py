import torch
import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
# import torch.nn.functional as F
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
from transformers import TransfoXLConfig, TransfoXLTokenizer, TransfoXLModel
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
                        help='Decide using chord or not.', default='./dictionary/dictionary_REMI-tempo-checkpoint.pkl')
    
    # validation opts
    parser.add_argument('--val_freq', type=int,
                        help='Validation frequency (every N epochs).', default=5)
    parser.add_argument('--early_stop_patience', type=int,
                        help='Early stopping patience.', default=5)
    
    # testing opts
    parser.add_argument('--prompt', type=int,
                        help='0 for generating from scratch, 1 for continue generating.', default=False)
    parser.add_argument('--prompt_path', type=str,
                        help='if prompt is True, you have to specify the continue generating midi file path.', default='')
        # './data/evaluation/000.midi'
    parser.add_argument('--n_target_bar', type=int,
                        help='Controll the generate result.', default=16)
    parser.add_argument('--temperature', type=float,
                        help='Controll the generate result.', default=1.2)
    parser.add_argument('--topk', type=int,
                        help='Controll the generate result.', default=5)
    parser.add_argument('--output_path', type=str,
                        help='output path', default='./results/from_scratch.midi')
    parser.add_argument('--model_path', type=str,
                        help='model path', default='./checkpoints/epoch_200.pkl')
    args = parser.parse_args()
    return args

class NewsDataset(Dataset):
    def __init__(self, train = True, midi_l = [], dict_pth = './dictionary/dictionary_REMI-tempo-checkpoint.pkl'):
        self.midi_l = midi_l
        self.tokenizer = REMI()
        self.checkpoint_path = dict_pth
        self.x_len = 512
        self.dictionary_path = dict_pth
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
        self.train = train
    
    def __len__(self):
        return len(self.parser)
    
    def __getitem__(self, index):
        if self.train:
            return self.parser[index]
        else:
            return self.words
            
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
        print(f'{len(midi_paths)} midis.')
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        print(f'{len(all_events)} events.')
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        # to training data
        print(f'{len(all_words)} words.')
        
        # 去掉group机制，直接创建序列对
        segments = []
        for words in all_words:
            if len(words) < self.x_len + 1:
                continue  # 跳过太短的序列
            
            # 创建重叠的序列对
            step_size = self.x_len // 2  # 50%重叠
            for i in range(0, len(words) - self.x_len, step_size):
                if i + self.x_len + 1 <= len(words):
                    x = words[i:i + self.x_len]
                    y = words[i + 1:i + self.x_len + 1]
                    segments.append([x, y])
        
        segments = np.array(segments)  # 形状: (N, 2, 512)
        print(segments.shape)
        return segments

class Model(nn.Module):
    def __init__(self, checkpoint, is_training=False):
        super(Model, self).__init__()
        # load dictionary
        self.dictionary_path = checkpoint
        self.checkpoint_path = checkpoint
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # model settings
        self.x_len = 512
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model

        self.configuration = TransfoXLConfig(   attn_type = 0,
                                                adaptive = False,
                                                clamp_len = -1,
                                                cutoffs = [],
                                                d_embed = self.d_embed,
                                                d_head = self.d_head,
                                                d_inner = self.d_ff,
                                                d_model = self.d_model,
                                                div_val = -1,
                                                dropatt = self.dropout,
                                                dropout = self.dropout,
                #                                 eos_token_id = ,
                                                init = 'normal',
                #                                 init_range = ,
                                                init_std = 0.02,
                                                layer_norm_epsilon = 0.001,
                                                mem_len = self.mem_len,
                                                n_head = self.n_head,
                                                n_layer = self.n_layer,
                                                pre_lnorm = 'normal',
                                                proj_init_std = 0.01,
                                                same_length = False,
                #                                 sample_softmax = ,
                                                tie_projs = [],
                                                vocab_size = self.n_token,
                                                untie_r = False)
         
        # Initializing a model (with random weights) from the configuration
        self.xl = TransfoXLModel(self.configuration)
        self.drop = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.d_embed, self.n_token)

    def forward(self, x):
        outputs = self.xl(input_ids = x)
        output = self.drop(outputs['last_hidden_state']) # dropout
        output_logit = self.linear(output)
        return output_logit

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
                
            # 去掉group循环，直接处理
            x = batch[:, 0, :].to(device).long()  # 输入序列
            y = batch[:, 1, :].to(device).long()  # 目标序列
            
            output_logit = model(x)
            
            # Calculate cross-entropy loss (which is negative log-likelihood)
            loss = nn.CrossEntropyLoss(reduction='sum')(output_logit.permute(0,2,1), y)
            
            total_nll += loss.item()
            total_tokens += y.numel()  # Total number of tokens
            batch_count += 1
    
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    
    model.train()  # Set back to training mode
    return avg_nll, perplexity, total_tokens

def temperature_sampling(logits, temperature, topk):
        # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        logits = torch.Tensor(logits)
        probs = nn.Softmax(dim=0)(logits / temperature)
        probs = np.array(probs)
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction
    
def test(prompt_path = './data/evaluation/000.midi', prompt = True, n_target_bar = 16,
         temperature = 1.2, topk = 5, output_path = '', model_path = ''):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # check path folder
    try:
        os.makedirs('./results', exist_ok=True)
        print("dir \'./results\' is created")
    except:
        pass

    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path)
        model = Model(checkpoint=opt.dict_path)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()

        test_data = NewsDataset(midi_l = [prompt_path], dict_pth = opt.dict_path)
        batch_size = 1
        
        # if prompt, load it. Or, random start
        if prompt:
            events = test_data.extract_events(prompt_path)
            words = [[test_data.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(test_data.event2word['Bar_None'])
        else:
            words = []
            for _ in range(batch_size):
                ws = [test_data.event2word['Bar_None']]
                if 'chord' in model.checkpoint_path:
                    tempo_classes = [v for k, v in test_data.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in test_data.event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in test_data.event2word.items() if 'Chord' in k]
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in test_data.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in test_data.event2word.items() if 'Tempo Value' in k]
                    ws.append(test_data.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        print('Start generating')
        while current_generated_bar < n_target_bar:
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = temp_x.cpu()
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])
            # model (prediction)
            temp_x = torch.Tensor(temp_x).long()
            temp_x = temp_x.to(device)
            # temp_x = (1_batch, 4_length)
            # print('temp_x shape =', temp_x.shape)
            output_logits = model(temp_x)
            # print('output_logits shape =', output_logits.shape)
            # output_logits = output_logits.permute(0,2,1)
            # sampling
            _logit = output_logits[0, -1].detach().cpu().numpy()
            # print('_logit shape =', _logit.shape)
            # break

            # print('_logit =',_logit.shape)
            word = temperature_sampling(
                logits=_logit, 
                temperature=temperature,
                topk=topk)

            words[0].append(word)

            # if bar event (only work for batch_size=1)
            if word == test_data.event2word['Bar_None']:
                current_generated_bar += 1
            # re-new mem
    #         batch_m = _new_mem
        # write
        if prompt:
            utils.write_midi(
                words=words[0][original_length:],
                word2event=test_data.word2event,
                output_path=output_path,
                prompt_path=prompt_path)
        else:
            utils.write_midi(
                words=words[0],
                word2event=test_data.word2event,
                output_path=output_path,
                prompt_path=None)
    
# train
def train(is_continue = False, checkpoints_path = ''):
    epochs = 200
    # create data list
    all_train_list = glob.glob('./data/train/*.midi')
    print('Total MIDI files:', len(all_train_list))
    
    # Split data into train and validation sets
    train_list, val_list = train_test_split(all_train_list, test_size=0.1, random_state=42)
    print('Train list len =', len(train_list))
    print('Validation list len =', len(val_list))
    
    # datasets
    train_dataset = NewsDataset(midi_l = train_list, dict_pth = opt.dict_path)
    val_dataset = NewsDataset(midi_l = val_list, dict_pth = opt.dict_path)
    
    # dataloaders
    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False)
    print('Dataloaders are created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # create model
    if not is_continue:
        start_epoch = 1
        model = Model(checkpoint=opt.dict_path).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=0.004*0.0002)
        
        # Initialize tracking variables
        best_val_nll = float('inf')
        patience_counter = 0
        train_nlls = []
        val_nlls = []
        
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model(checkpoint=opt.dict_path).to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
        optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400000, eta_min=0.004*0.0002)
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load tracking variables
        best_val_nll = checkpoint.get('best_val_nll', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        train_nlls = checkpoint.get('train_nlls', [])
        val_nlls = checkpoint.get('val_nlls', [])

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    try:
        os.makedirs('./checkpoints', exist_ok=True)
        print("dir is created")
    except:
        pass
    
    for epoch in range(start_epoch, epochs+1):
        # Training phase
        single_epoch = []
        for i in tqdm(train_dataloader, desc=f"Epoch {epoch} Training"):
            # 去掉group循环，直接处理
            x = i[:, 0, :].to(device).long()  # 输入序列 (batch, 512)
            y = i[:, 1, :].to(device).long()  # 目标序列 (batch, 512)
            
            output_logit = model(x)
            # output_logit = (batch, 512, vocab_size)
            
            loss = nn.CrossEntropyLoss()(output_logit.permute(0,2,1), y)
            loss.backward()
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Calculate training NLL for this epoch
        train_epoch_nll = np.array(single_epoch).mean()
        losses.append(train_epoch_nll)
        train_nlls.append(train_epoch_nll)
        
        print(f'>>> Epoch: {epoch}, Train NLL: {train_epoch_nll:.5f}')
        
        # Validation phase
        if epoch % opt.val_freq == 0 or epoch == epochs:
            print(f"Calculating validation NLL for epoch {epoch}...")
            val_nll, val_perplexity, val_tokens = calculate_nll(model, val_dataloader, device, max_batches=50)
            val_nlls.append(val_nll)
            
            print(f'>>> Epoch: {epoch}, Validation NLL: {val_nll:.5f}, Perplexity: {val_perplexity:.2f}')
            print(f'>>> Evaluated on {val_tokens} tokens')
            
            # Early stopping check
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                patience_counter = 0
                print(f'>>> New best validation NLL: {best_val_nll:.5f}')
                # Save best model
                best_model_path = './checkpoints/best_model.pkl'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_nll': train_epoch_nll,
                    'val_nll': val_nll,
                    'best_val_nll': best_val_nll,
                    'patience_counter': patience_counter,
                    'train_nlls': train_nlls,
                    'val_nlls': val_nlls,
                }, best_model_path)
                print(f'>>> Best model saved to {best_model_path}')
            else:
                patience_counter += 1
                print(f'>>> No improvement. Patience: {patience_counter}/{opt.early_stop_patience}')
                
                if patience_counter >= opt.early_stop_patience:
                    print(f'>>> Early stopping triggered after {patience_counter} epochs without improvement')
                    break
        
        # Save regular checkpoint
        checkpoint_path = f'./checkpoints/epoch_{epoch:03d}.pkl'
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_nll': train_epoch_nll,
            'val_nll': val_nlls[-1] if val_nlls else float('inf'),
            'best_val_nll': best_val_nll,
            'patience_counter': patience_counter,
            'train_nlls': train_nlls,
            'val_nlls': val_nlls,
        }, checkpoint_path)
        
        # Save training history
        np.save('training_nlls_history.npy', np.array(train_nlls))
        np.save('validation_nlls_history.npy', np.array(val_nlls))
        
        # Print NLL summary every 10 epochs
        if epoch % 10 == 0:
            print("\n" + "="*80)
            print(f"EPOCH {epoch} SUMMARY:")
            print(f"Train NLL: {train_epoch_nll:.5f}")
            if val_nlls:
                print(f"Best Validation NLL: {best_val_nll:.5f}")
                print(f"Current Validation NLL: {val_nlls[-1]:.5f}")
            print(f"Patience Counter: {patience_counter}/{opt.early_stop_patience}")
            print("="*80 + "\n")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print(f"Total epochs trained: {epoch}")
    print(f"Best validation NLL: {best_val_nll:.5f}")
    print(f"Final train NLL: {train_nlls[-1]:.5f}")
    if val_nlls:
        print(f"Final validation NLL: {val_nlls[-1]:.5f}")
    print("="*80)
    
    losses = np.array(losses)
    np.save('training_losses_final.npy', losses)

def main(opt):

    # train
    if opt.is_train:
        # train from scratch
        if not opt.is_continue:
            train()
        # continue training
        else:
            train(is_continue = opt.is_continue, checkpoints_path = opt.continue_pth)

    else:
        # generate from screatch
        if not opt.prompt:
            test(prompt = opt.prompt, n_target_bar = opt.n_target_bar, temperature = opt.temperature, topk = opt.topk, 
                output_path = opt.output_path, model_path = opt.model_path)
        
        # continue generate
        else:
            test(prompt_path = opt.prompt_path, n_target_bar = opt.n_target_bar, temperature = opt.temperature, topk = opt.topk,
                output_path = opt.output_path, model_path = opt.model_path)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)