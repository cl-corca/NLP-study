import nltk.translate.bleu_score as bleu
import torch
from torch import IntTensor

from model import Transformer
from datatool import get_test_dataloader, get_tokenizer
from constants import *

def predict(model, tokenizer, sentence, max_sequence_length) -> str:
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
    pad_idx = PAD_IDX

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
    cnt = 0
    len_loader = len(dataloader)

    for i, (en, de) in enumerate(dataloader):
        for e, d in zip(list(en), list(de)):
            output = predict(model, tokenizer, e, max_sequence_length=MAX_SEQUENCE_LENGTH)
            total_bleu += bleu.sentence_bleu([d], output)
            cnt += 1
        print(f"BLEU: {total_bleu/cnt} {i} / {len_loader}")
    print(f"BLEU: {total_bleu/cnt}")


def translate_examples():
    eng_examples = [
        "The quick brown fox jumps over the lazy dog.",
        "I enjoy listening to classical music while I work.",
        "The sunset painted the sky in shades of pink and orange.", 
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

    for e in eng_examples:
        print("=================================")
        print(e)
        print(predict(model, tokenizer, e, max_sequence_length=MAX_SEQUENCE_LENGTH))


if __name__ == "__main__":
    test()
    #ranslate_examples()
