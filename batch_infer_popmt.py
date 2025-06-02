#!/usr/bin/env python3
"""
Optimized single GPU batch generation script
Maximum efficiency for single GPU setup
"""

import torch
import numpy as np
import pickle
import os
import time
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your original modules
import utils
from torch import nn
from transformers import TransfoXLConfig, TransfoXLModel

class Model(nn.Module):
    def __init__(self, checkpoint, is_training=False):
        super(Model, self).__init__()
        self.dictionary_path = checkpoint
        self.checkpoint_path = checkpoint
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        
        # Model settings
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

        self.configuration = TransfoXLConfig(   
            attn_type=0, adaptive=False, clamp_len=-1, cutoffs=[],
            d_embed=self.d_embed, d_head=self.d_head, d_inner=self.d_ff,
            d_model=self.d_model, div_val=-1, dropatt=self.dropout,
            dropout=self.dropout, init='normal', init_std=0.02,
            layer_norm_epsilon=0.001, mem_len=self.mem_len,
            n_head=self.n_head, n_layer=self.n_layer,
            pre_lnorm='normal', proj_init_std=0.01, same_length=False,
            tie_projs=[], vocab_size=self.n_token, untie_r=False
        )
         
        self.xl = TransfoXLModel(self.configuration)
        self.drop = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.d_embed, self.n_token)

    def forward(self, x):
        outputs = self.xl(input_ids=x)
        output = self.drop(outputs['last_hidden_state'])
        output_logit = self.linear(output)
        return output_logit

def temperature_sampling(logits, temperature, topk, device):
    """GPU优化的采样函数"""
    logits = logits / temperature
    
    if topk == 1:
        return torch.argmax(logits).item()
    else:
        # 在GPU上进行top-k采样
        values, indices = torch.topk(logits, topk)
        probs = torch.softmax(values, dim=0)
        
        # 在GPU上采样
        idx = torch.multinomial(probs, 1)
        return indices[idx].item()

class BatchMusicGenerator:
    def __init__(self, model_path, dict_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 一次性加载模型和字典
        self.model, self.event2word, self.word2event = self._load_model(model_path, dict_path)
        
        # 预计算常用的token集合
        self._precompute_tokens()
        
    def _load_model(self, model_path, dict_path):
        """加载模型和字典"""
        print("Loading model and dictionary...")
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        model = Model(checkpoint=dict_path)
        model.load_state_dict(checkpoint['model'])
        model = model.to(self.device)
        model.eval()
        
        # 关闭梯度计算以节省内存
        for param in model.parameters():
            param.requires_grad = False
        
        # 加载字典
        event2word, word2event = pickle.load(open(dict_path, 'rb'))
        
        print(f"Model loaded successfully! Vocab size: {len(event2word)}")
        return model, event2word, word2event
    
    def _precompute_tokens(self):
        """预计算常用token，提高效率"""
        self.bar_token = self.event2word['Bar_None']
        self.position_token = self.event2word['Position_1/16']
        
        # 预计算tempo相关tokens
        self.tempo_classes = torch.tensor([
            v for k, v in self.event2word.items() if 'Tempo Class' in k
        ], device=self.device)
        
        self.tempo_values = torch.tensor([
            v for k, v in self.event2word.items() if 'Tempo Value' in k
        ], device=self.device)
        
        # 检查是否有chord
        self.has_chord = 'chord' in self.model.checkpoint_path.lower()
        if self.has_chord:
            self.chord_tokens = torch.tensor([
                v for k, v in self.event2word.items() if 'Chord' in k
            ], device=self.device)
    
    def _initialize_sequence(self):
        """初始化生成序列"""
        words = [self.bar_token, self.position_token]
        
        if self.has_chord:
            # 随机选择chord
            chord_idx = torch.randint(0, len(self.chord_tokens), (1,), device=self.device)
            words.append(self.chord_tokens[chord_idx].item())
            words.append(self.position_token)
        
        # 随机选择tempo
        tempo_class_idx = torch.randint(0, len(self.tempo_classes), (1,), device=self.device)
        tempo_value_idx = torch.randint(0, len(self.tempo_values), (1,), device=self.device)
        
        words.append(self.tempo_classes[tempo_class_idx].item())
        words.append(self.tempo_values[tempo_value_idx].item())
        
        return words
    
    def generate_single(self, n_target_bar=16, temperature=1.2, topk=5, seed=None):
        """生成单首音乐（GPU优化版本）"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        
        words = self._initialize_sequence()
        original_length = len(words)
        current_generated_bar = 0
        
        sequence = torch.tensor(words, device=self.device, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            while current_generated_bar < n_target_bar:
                output_logits = self.model(sequence)
                
                last_logits = output_logits[0, -1]
                
                next_token = temperature_sampling(last_logits, temperature, topk, self.device)
                
                next_token_tensor = torch.tensor([[next_token]], device=self.device, dtype=torch.long)
                sequence = torch.cat([sequence, next_token_tensor], dim=1)
                
                if next_token == self.bar_token:
                    current_generated_bar += 1
                
                if sequence.size(1) > 1024:
                    sequence = sequence[:, -512:]
        
        # 返回生成的序列（去除初始部分）
        generated_tokens = sequence[0, original_length:].cpu().numpy().tolist()
        return generated_tokens
    
    def batch_generate(self, num_pieces, n_target_bar=16, temperature=1.2, 
                      topk=5, output_dir='./gpu_batch_results', 
                      save_interval=10, seed=42):
        """批量生成音乐"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting batch generation on {self.device}")
        print(f"Target: {num_pieces} pieces, {n_target_bar} bars each")
        print(f"Temperature: {temperature}, Top-k: {topk}")
        print("-" * 60)
        
        # 统计信息
        total_time = 0
        successful = 0
        failed = 0
        
        # 生成循环
        pbar = tqdm(range(num_pieces), desc="Generating")
        
        for i in pbar:
            start_time = time.time()
            
            try:
                # 生成音乐
                generated_tokens = self.generate_single(
                    n_target_bar=n_target_bar,
                    temperature=temperature,
                    topk=topk,
                    seed=seed + i
                )
                
                # 保存MIDI文件
                output_path = os.path.join(output_dir, f"generated_{i:06d}.midi")
                utils.write_midi(
                    words=generated_tokens,
                    word2event=self.word2event,
                    output_path=output_path,
                    prompt_path=None
                )
                
                successful += 1
                generation_time = time.time() - start_time
                total_time += generation_time
                
                # 更新进度条
                avg_time = total_time / (i + 1)
                pbar.set_postfix({
                    'Success': successful,
                    'Failed': failed,
                    'Avg Time': f'{avg_time:.2f}s',
                    'ETA': f'{avg_time * (num_pieces - i - 1):.0f}s'
                })
                
            except Exception as e:
                failed += 1
                print(f"\nError generating piece {i}: {e}")
                
            # 定期清理GPU缓存
            if (i + 1) % save_interval == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # 最终统计
        print("\n" + "="*60)
        print("BATCH GENERATION COMPLETED!")
        print(f"Successful: {successful}/{num_pieces}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/num_pieces*100:.1f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per piece: {total_time/successful:.2f} seconds" if successful > 0 else "N/A")
        print(f"Output directory: {output_dir}")
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='Single GPU Batch Music Generation')
    
    # Model settings
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dict_path', type=str, required=True,
                        help='Path to dictionary file')
    
    # Generation settings
    parser.add_argument('--num_pieces', type=int, default=500,
                        help='Number of pieces to generate')
    parser.add_argument('--n_target_bar', type=int, default=16,
                        help='Number of bars to generate per piece')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Sampling temperature')
    parser.add_argument('--topk', type=int, default=10,
                        help='Top-k sampling parameter')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./generated_midis_popmt',
                        help='Output directory for generated MIDI files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for GPU cache cleanup')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    generator = BatchMusicGenerator(
        model_path=args.model_path,
        dict_path=args.dict_path,
        device=args.device
    )
    
    generator.batch_generate(
        num_pieces=args.num_pieces,
        n_target_bar=args.n_target_bar,
        temperature=args.temperature,
        topk=args.topk,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        seed=args.seed
    )

if __name__ == '__main__':
    main()