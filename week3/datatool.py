import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.implementations import CharBPETokenizer
from torch import IntTensor, LongTensor, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from constants import *

def load_data(train, num_data=None) -> tuple[list[str], list[str]]:
    """
    Download raw train.de and train.en from https://nlp.stanford.edu/projects/nmt/
    Args: 
        train: bool 
        num_data: int 
    Return: 
        tuple of list(english_sentences, german_sentences)
    """
    if train:
        f_en = open(os.path.join(PATH_DATA, "train.en"))
        f_de = open(os.path.join(PATH_DATA, "train.de"))
    else:
        f_en = open(os.path.join(PATH_DATA, "newstest2014.en"))
        f_de = open(os.path.join(PATH_DATA, "newstest2014.de"))
    en = f_en.readlines()
    if num_data is not None:
        en = en[:num_data]
    de = f_de.readlines()
    if num_data is not None:
        de = de[:num_data]
    f_en.close()
    f_de.close()
    en = [sentence.strip() for sentence in en]
    de = [sentence.strip() for sentence in de]
    return en, de

def save_tokenizer():
    tokenizer = CharBPETokenizer(lowercase=True, suffix="")
    tokenizer.train(
        files=[
            os.path.join(PATH_DATA, "train.de"),
            os.path.join(PATH_DATA, "train.en"),
        ],
        vocab_size=47000,
        min_frequency=1,
        special_tokens=SPECIAL_SYMBOLS,
        suffix="",
    )
    tokenizer.save(os.path.join(PATH_DATA, "tokenizer.json"))

def get_tokenizer(): 
    saved_tokenizer_path = os.path.join(PATH_DATA, "tokenizer.json")
    if not os.path.exists(saved_tokenizer_path):
        save_tokenizer()
    tokenizer = Tokenizer.from_file(saved_tokenizer_path)
    return tokenizer

def get_dataloader(batch_size=BATCH_SIZE):
    tokenizer = get_tokenizer()
    train_en, train_de = load_data(True)
    dataset = WMT14Dataset(
        tokenizer,
        train_en,
        train_de,
        seq_length=MAX_SEQUENCE_LENGTH,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def get_test_dataloader(batch_size=32):
    tokenizer = get_tokenizer()
    test_en, test_de = load_data(False)
    dataset = WMT14TestDataset(
        tokenizer,
        test_en,
        test_de,
        seq_length=100,
    )    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader

class WMT14Dataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        train_en: list[str],
        train_de: list[str],
        seq_length: int = 100,
    ):
        super(WMT14Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.en = train_en
        self.de = train_de
        self.seq_length = seq_length
        self.cached_en = [None for _ in range(len(self.en))]
        self.cached_de = [None for _ in range(len(self.de))]
        self.cached_label = [None for _ in range(len(self.de))]
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.unk = self.tokenizer.token_to_id("<unk>")
        self.pad = self.tokenizer.token_to_id("<pad>")
        self.sos = self.tokenizer.token_to_id("<sos>")
        self.eos = self.tokenizer.token_to_id("<eos>")
        self.mask = self.tokenizer.token_to_id("<mask>")

        if len(self.en) != len(self.de):
            raise ValueError("train_en and train_de must have same length.")

        self.pad_one_hot = nn.functional.one_hot(
            LongTensor([self.pad]), num_classes=self.vocab_size
        )

    def __getitem__(self, index: int) -> tuple[IntTensor, IntTensor, LongTensor]:
        if self.cached_en[index] is not None:
            return (
                self.cached_en[index],
                self.cached_de[index],
                self.cached_label[index],
            )

        en_sentence = self.en[index]
        de_sentence = self.de[index]
        en_sentence = self.tokenizer.encode(en_sentence).ids
        de_sentence = self.tokenizer.encode(de_sentence).ids
        en_sentence = en_sentence[: self.seq_length - 2]
        de_sentence = de_sentence[: self.seq_length - 2]
        en_sentence = [self.sos] + en_sentence + [self.eos]
        de_sentence = [self.sos] + de_sentence + [self.eos]
        if len(en_sentence) < self.seq_length:
            en_sentence += [self.pad] * (self.seq_length - len(en_sentence))
        if len(de_sentence) < self.seq_length:
            de_sentence += [self.pad] * (self.seq_length - len(de_sentence))

        en_sentence = IntTensor(en_sentence)
        de_sentence = IntTensor(de_sentence)
        label = de_sentence[1:].to(torch.int64)
        label = torch.cat([label, LongTensor([self.pad])], dim=0)
        self.cached_en[index] = en_sentence
        self.cached_de[index] = de_sentence
        self.cached_label[index] = label

        return en_sentence, de_sentence, label

    def __len__(self):
        return len(self.en)

class WMT14TestDataset(WMT14Dataset):
    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.en[index], self.de[index]

# test code to save tokenizer.json to data directory
if __name__ == '__main__':
    print("run get_tokenizer")
    get_tokenizer()
    print("run get_dataloader")
    dataloader = get_dataloader()
    print(dataloader) 
    print("run get_test_dataloader")
    dataloader2 = get_test_dataloader()
    print(dataloader2) 
