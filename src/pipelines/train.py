from torch.utils.data import DataLoader

from dataset.splits import datasets_get, splits_get
from src.config.vars import BATCH_SIZE, NUM_WORKERS
from src.dataset.collates import customized_collate_fn
from src.dataset.data import INSTRUCTION_DATA

splits = splits_get(INSTRUCTION_DATA)
datasets = datasets_get(splits)


def loader_get(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )


train_loader = loader_get(datasets["train"])
test_loader = loader_get(datasets["test"])
eval_loader = loader_get(datasets["eval"])
