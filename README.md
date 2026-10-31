# Transformer from scratch in PyTorch

A from-scratch implementation of the encoder–decoder Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
trained on English → French machine translation.

Every building block (embeddings, positional encoding, multi-head attention, layer norm,
feed-forward, residual connections, encoder/decoder stacks) is written with plain
`torch.nn` modules, with no use of `nn.Transformer`.

## Features

- Full encoder–decoder architecture with configurable depth, width and number of heads
- Sinusoidal positional encoding and scaled embeddings (`× √d_model`)
- Multi-head scaled dot-product attention with padding and causal masks
- Self-attention, cross-attention and position-wise feed-forward sub-layers
- Residual connections with layer normalization (pre-norm variant)
- Xavier-uniform parameter initialization
- Word-level tokenizers built with Hugging Face `tokenizers`
- Training loop with label smoothing, per-epoch checkpoints and resume support
- Greedy decoding with example translations printed after every epoch
- TensorBoard logging of the training loss

## Architecture

| Component | Class | Notes |
|---|---|---|
| Token embeddings | `InputEmbeddings` | `nn.Embedding` scaled by `√d_model` |
| Positional encoding | `PositionalEncoding` | Fixed sin/cos encoding, stored as a buffer, followed by dropout |
| Layer norm | `LayerNormalization` | Learnable scale and bias |
| Feed-forward | `FeedForwardBlock` | `Linear → ReLU → Dropout → Linear` |
| Attention | `MultiHeadAttentionBlock` | `h` heads, `d_k = d_model / h`, masks applied before softmax |
| Residual | `ResidualConnection` | `x + Dropout(sublayer(LayerNorm(x)))` |
| Encoder | `EncoderBlock`, `Encoder` | Self-attention + feed-forward, final layer norm |
| Decoder | `DecoderBlock`, `Decoder` | Masked self-attention + cross-attention + feed-forward, final layer norm |
| Output | `ProjectionLayer` | Linear to vocabulary size followed by `log_softmax` |
| Full model | `Transformer` | `encode`, `decode` and `project` methods |

The model is assembled with `build_transformer(...)`.

Default hyperparameters (as in the base model of the paper):

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `N` (layers per stack) | 6 |
| `h` (attention heads) | 8 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |

## Training setup

- **Dataset:** [`Helsinki-NLP/opus_books`](https://huggingface.co/datasets/Helsinki-NLP/opus_books), English–French pair
- **Split:** 90 % train / 10 % validation (random)
- **Tokenization:** word-level tokenizer per language (whitespace pre-tokenizer, `min_frequency=2`),
  special tokens `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`; tokenizers are cached as `tokenizer_{lang}.json`
- **Optimizer:** Adam, `lr = 1e-4`, `eps = 1e-9`
- **Loss:** cross-entropy with label smoothing 0.1, padding ignored
- **Validation:** greedy decoding on a few sentences, printing source, target and prediction

## Configuration

All settings are in `get_config()`:

| Key | Default | Description |
|---|---|---|
| `batch_size` | 4 | Training batch size |
| `num_epochs` | 20 | Number of epochs |
| `lr` | `1e-4` | Learning rate |
| `seq_len` | 485 | Maximum sequence length (sentences are padded to this length) |
| `d_model` | 512 | Model dimension |
| `lang_src` / `lang_tgt` | `en` / `fr` | Source and target languages |
| `model_folder` | `weights` | Directory for checkpoints |
| `tokenizer_file` | `tokenizer_{0}.json` | Tokenizer cache path |
| `preload` | `None` | Epoch of a checkpoint to resume from (e.g. `"05"`) |
| `experiment_name` | `runs/tmodel` | TensorBoard log directory |

## Installation

```bash
pip install torch datasets tokenizers tensorboard tqdm
```

A CUDA-capable GPU is recommended; the script falls back to CPU if none is available.

## Usage

Train:

```bash
python transformer.py
```

A checkpoint (model weights, optimizer state, epoch and global step) is saved in `weights/`
at the end of every epoch.

Resume from a checkpoint by setting `preload` in `get_config()`, for example `"preload": "05"`.

Monitor training:

```bash
tensorboard --logdir runs
```

### Using the model directly

```python
from transformer import build_transformer

model = build_transformer(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    src_seq_len=485,
    tgt_seq_len=485,
    d_model=512,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=2048,
)

# encoder_output = model.encode(src, src_mask)
# decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
# log_probs      = model.project(decoder_output)
```

## Repository layout

```
transformer.py   # model, dataset, tokenizers, training loop, greedy decoding, config
```

## Reference

- A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
