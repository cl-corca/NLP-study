import torch
from functools import partial
from torch import nn, optim 
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from constants import (
    BATCH_SIZE,
    CBOW_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
    MAX_TOKEN,
) 

def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(ds_type, vocab=None):
    
    #CL: Choose one of WikiText2 and WikiText103
    #data_iter = to_map_style_dataset(WikiText2(root='data/', split=ds_type)) 
    data_iter = to_map_style_dataset(WikiText103(root='data/', split=ds_type)) 

    tokenizer = get_tokenizer("basic_english", language="en")
    if not vocab:
        vocab = build_vocab_from_iterator(
            map(tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=MIN_WORD_FREQUENCY,
            max_tokens=MAX_TOKEN, 
        )
        vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    dataloader = DataLoader(
        data_iter,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline),
    )
    return dataloader, vocab

# test code
if __name__ == '__main__':
    train_dataloader, vocab = get_dataloader_and_vocab(ds_type='train')
    print(f"vocabulary size: {len(vocab.get_stoi())}")

