from dataset.instruction import InstructionDataset


def splits_get(data, p_train=0.85, p_test=0.1, p_eval=0.05):
    size = len(data)
    train_range = int(size * p_train)
    test_range = int(size * p_test) + train_range

    return {
        "train": data[:train_range],
        "test": data[train_range:test_range],
        "eval": data[test_range:],
    }


def datasets_get(splits):
    return {
        "train": InstructionDataset(splits["train"]),
        "test": InstructionDataset(splits["test"]),
        "eval": InstructionDataset(splits["eval"]),
    }
