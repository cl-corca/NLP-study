import os
from functools import partial
from random import randint

from datasets import load_dataset
from tokenizers import Tokenizer
from torch import IntTensor, LongTensor
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from constants import *

PATH_DATA = os.path.join(os.path.dirname(__file__), "data")
special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>", "<mask>"]


def load_data():
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(PATH_DATA, "train"),
    )
    print(dataset["train"])

    for data in dataset:
        print(data)
        break


def save_tokenizer():
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(PATH_DATA, "train"),
    )["train"]

    def iter_dataset():
        for data in dataset:
            yield data["text"]

    tokenizer.train_from_iterator(
        iterator=iter_dataset(),
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=special_tokens,
    )
    tokenizer.save(os.path.join(PATH_DATA, "tokenizer.json"))


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(os.path.join(PATH_DATA, "tokenizer.json"))
    return tokenizer

def get_tokenizer():
    saved_tokenizer_path = os.path.join(PATH_DATA, "tokenizer.json")
    if not os.path.exists(saved_tokenizer_path):
        save_tokenizer()
    tokenizer = Tokenizer.from_file(saved_tokenizer_path)
    return tokenizer

def collate_fn(
    batch: list[dict[str, str]], tokenizer: Tokenizer
) -> tuple[IntTensor, LongTensor]:
    batch_texts = [data["text"] for data in batch]
    batch_tokens = tokenizer.encode_batch(batch_texts)

    batch_inputs, batch_labels = [], []

    for tokens in batch_tokens:
        tokens = tokens.ids
        if len(tokens) <= BLOCK_SIZE + 1:
            tokens += [tokenizer.token_to_id("<pad>")] * (
                BLOCK_SIZE + 1 - len(tokens)
            )
            batch_inputs.append(tokens[:-1])
            batch_labels.append(tokens[1:])
            continue
        idx = randint(0, len(tokens) - BLOCK_SIZE - 1)
        batch_inputs.append(tokens[idx : idx + BLOCK_SIZE])
        batch_labels.append(tokens[idx + 1 : idx + BLOCK_SIZE + 1])

    return IntTensor(batch_inputs), LongTensor(batch_labels)

def get_dataloader(batch_size=BATCH_SIZE):
    tokenizer = get_tokenizer()
    dataset = load_dataset(
        "Skylion007/openwebtext",
        cache_dir=os.path.join(PATH_DATA, "train"),
    )
    dataloader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
        shuffle=True,
    )
    return dataloader


if __name__ == "__main__":
    get_tokenizer()
