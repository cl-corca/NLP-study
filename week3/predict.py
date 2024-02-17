import nltk.translate.bleu_score as bleu
import torch
from tokenizers import Tokenizer
from torch import IntTensor, nn
from torch.utils.data import DataLoader

from model import Transformer
from constants import *
from datatool import WMT14Dataset, load_data, get_test_dataloader, get_tokenizer


class WMT14TestDataset(WMT14Dataset):
    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.en[index], self.de[index]


def predict(model: nn.Module, tokenizer: Tokenizer, sentence: str, max_sequence_length: int) -> str:
    model.eval()

    tokens = tokenizer.encode(sentence).ids

    tokens = tokens[: max_sequence_length - 2]
    tokens = (
        [tokenizer.token_to_id("<sos>")] + tokens + [tokenizer.token_to_id("<eos>")]
    )
    if len(tokens) < max_sequence_length:
        tokens += [tokenizer.token_to_id("<pad>")] * (max_sequence_length - len(tokens))

    encoder_input = IntTensor([tokens])
    decoder_input = IntTensor(
        [
            [tokenizer.token_to_id("<sos>")]
            + [tokenizer.token_to_id("<pad>")] * (max_sequence_length - 1)
        ]
    )

    for i in range(max_sequence_length):
        with torch.no_grad():
            outputs = model(encoder_input, decoder_input)
            predicted_tokens = outputs[0].argmax(dim=-1)
            predicted_token = predicted_tokens[i].item()

            if predicted_token == tokenizer.token_to_id("<eos>"):
                break
            decoder_input[:, i + 1] = predicted_token

    decoder_output = decoder_input[0][1:].tolist()

    return " ".join(
        tokenizer.id_to_token(token)
        for token in decoder_output
        if token != tokenizer.token_to_id("<pad>")
        and token != tokenizer.token_to_id("<eos>")
    )


def test():
    tokenizer = get_tokenizer()
    dataloader = get_test_dataloader()
    pad_idx = PAD_IDX #dataset.pad

    #pad_idx = dataset.pad

    model = Transformer(
        pad_idx, #TODO (cl): check PAD_IDX 
        VOCAB_SIZE, 
        D_MODEL, 
        NUM_HEADS, 
        NUM_LAYERS, 
        D_FF, 
        MAX_SEQUENCE_LENGTH, 
        DROPOUT
    )

    model.load_state_dict(torch.load("data/transformer_model_final.json"))

    total_bleu = 0.0
    count = 0

    for i, (en, de) in enumerate(dataloader):
        en = list(en)
        de = list(de)
        for en_, de_ in zip(en, de):
            output = predict(model, tokenizer, en_, max_sequence_length=MAX_SEQUENCE_LENGTH)
            total_bleu += bleu.sentence_bleu([de_], output)
            count += 1
            print(count)
        print(f"BLEU: {total_bleu / count} {i} / {len(dataloader)}")

    print(f"BLEU: {total_bleu / count}")


def generate_examples():
    en = [
        "The quick brown fox jumps over the lazy dog.",
        "Every morning, I enjoy a cup of coffee while watching the sunrise.",
        "Technology is rapidly advancing, transforming how we live and work.",
    ]

    tokenizer = get_tokenizer()
    pad_idx = tokenizer.token_to_id("<pad>")
    model = Transformer(
        pad_idx, #TODO (cl): check PAD_IDX 
        VOCAB_SIZE, 
        D_MODEL, 
        NUM_HEADS, 
        NUM_LAYERS, 
        D_FF, 
        MAX_SEQUENCE_LENGTH, 
        DROPOUT
    )
    model.load_state_dict(torch.load("data/transformer_model_final.json"))

    for _en in en:
        print("=================================")
        print(_en)
        print(predict(model, tokenizer, _en, max_sequence_length=MAX_SEQUENCE_LENGTH))


if __name__ == "__main__":
    #test()
    generate_examples()
