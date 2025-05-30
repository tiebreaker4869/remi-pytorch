# remi-pytorch
Implement the pop-music transformer and integrate with hugging face API

# Data preparation
Please follow the steps in REMI. https://github.com/YatingMusic/remi

# Patch Fix
You may need to apply [this patch](https://github.com/ellemcfarlane/transformers/commit/29739e690bb3448e68914173fea56c1974fba1cb) if using recent versions of transformers library.

# Training
You can change `--dict_path` to specify whether using chord or not.

## from scratch
`python main.py --dict_path your-path-here`

## continue
```
python main.py \
--is_continue 1 \
--continue_pth your-path-here
```

# Testing
`--dict_path` should be same as training option.

## generate from scratch
```
python main.py \
--is_train 0 \
--prompt 0 \
--output_path your-path-here \
--model_path your-path-here
```
                
## continue generating
```
python main.py --is_train 0 \
--prompt_path your-path-here \
--output_path your-path-here \
--model_path your-path-here
```