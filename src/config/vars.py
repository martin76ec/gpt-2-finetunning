NUM_WORKERS = 4
NUM_EPOCHS = 5
LR = 5e-5
WD = 0.1

BATCH_SIZE = 8

DEVICE = "cuda"

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}
