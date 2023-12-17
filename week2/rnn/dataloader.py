import torch
from functools import partial
from torch import nn, optim 
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

MIN_WORD_FREQUENCY = 1
MAX_WORDS = 25
BATCH_SIZE = 1024

def get_dataloader_and_vocab(ds_type, vocab=None, classes=None):
    """Returns dataloader and vocab. 

    Args: 
        ds_type: 'train' or 'val' 
        vocab: given or None 
    
    Returns:
        dataloader
        vocab
        classes
    """
    data_set = to_map_style_dataset(AG_NEWS(split=ds_type))
    tokenizer = get_tokenizer("basic_english", language="en")

    def _yield_tokens(data_set):
        for _, text in data_set:
            yield tokenizer(text)

    if not vocab:
        vocab = build_vocab_from_iterator(
            _yield_tokens(data_set),
            specials=["<unk>"],
            min_freq=MIN_WORD_FREQUENCY,
        )
        vocab.set_default_index(vocab["<unk>"])
    
    if not classes:
        classes = set([label for (label, text) in data_set])

    def _vectorize_batch(batch):
        Y, X = list(zip(*batch))
        X = [vocab(tokenizer(text)) for text in X]
        X = [tokens+([0]* (MAX_WORDS-len(tokens))) if len(tokens)<MAX_WORDS else tokens[:MAX_WORDS] for tokens in X] ## Bringing all samples to max_words length.
        return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]

    dataloader = DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_vectorize_batch,
    )
    return dataloader, vocab, classes

# test code
if __name__ == '__main__':
    train_loader, vocab, classes = get_dataloader_and_vocab(ds_type='train')
    print(train_loader)
    print(f"train vocabulary size: {len(vocab)}")
    print(classes)
    test_loader, vocab, classes = get_dataloader_and_vocab(ds_type='test', vocab=vocab, classes=classes)
    print(f"test vocabulary size: {len(vocab)}")
    print(classes)

