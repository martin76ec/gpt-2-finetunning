import time

import tiktoken
import torch

from src.config.vars import DEVICE, GPT_CONFIG_124M, LR, NUM_EPOCHS, WD
from src.dataset.data import INSTRUCTION_DATA
from src.dataset.instruction import format_input
from src.dataset.splits import splits_get
from src.models.gpt_2 import GPTModel
from src.pipelines.prepare_data import eval_loader, train_loader
from src.pipelines.train import train_model_simple

start_time = time.time()
torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("data/gpt2-small-124M.pth"))
model.eval()
model.to(DEVICE)

splits = splits_get(INSTRUCTION_DATA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


tokenizer = tiktoken.get_encoding("gpt2")

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    eval_loader,
    optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(splits["eval"][0]),
    tokenizer=tokenizer,
)

torch.save(model.state_dict(), "outputs/gpt2-tunned.pth")
