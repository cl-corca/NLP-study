import math

import torch
from tokenizers import Tokenizer
from torch import IntTensor, nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import WikiText103, WikiText2

from constants import *
from model import GPT
from datatool import get_tokenizer


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, stride: int = 512):
        self.stride = stride
        _train, test = WikiText103(root=PATH_DATA, split=("train", "test"))
        self.encodings = IntTensor(tokenizer.encode("\n\n".join(test)).ids)

    def __getitem__(self, index: int):
        return (
            self.encodings[index * self.stride : (index + 1) * self.stride],
            self.encodings[
                index * self.stride + 1 : (index + 1) * self.stride + 1
            ].type(torch.LongTensor),
        )

    def __len__(self):
        return (len(self.encodings) // self.stride) - 1


def test():
    tokenizer = get_tokenizer()

    dataset = WikiTextDataset(tokenizer=tokenizer, stride=BLOCK_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer),
        # num_workers=4,
        # shuffle=True,
    )

    model = GPT()
    model.load_state_dict(torch.load("data/model.v2.epoch_0.step_95000"))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    #model.to(config.device)
    model.to("cuda:1") #TODO (cl): to avoid conflict, hard coded cuda:1 not constants.DEVICE
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to("cuda:1")
            label = label.to("cuda:1")
            pred = model(data)
            pred = torch.transpose(pred, 1, 2)
            loss = criterion(pred, label)
            losses.append(loss.item())
            print(loss)

    print(f"perplexity: {math.exp(sum(losses) / (len(losses)))}")


if __name__ == "__main__":
    test()