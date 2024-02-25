import torch
from torch import IntTensor

from constants import *
from model import GPT
from datatool import load_tokenizer


def load_model():
    model = GPT()
    model.load_state_dict(torch.load("data/model.v2.epoch_0.step_95000"))
    model.to(DEVICE)
    model.eval()
    return model


def generate(model: GPT, prompt: str, max_iter: int = 100):
    tokenizer = load_tokenizer()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = IntTensor([input_ids])
    output_ids = model.generate(input_ids, max_iter=max_iter, temperature=1.0, top_k=40)
    output_ids = output_ids.tolist()
    print(tokenizer.decode(output_ids[0]))
    return output_ids


def stream(model: GPT, prompt: str, max_iter: int = 100):
    tokenizer = load_tokenizer()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = IntTensor([input_ids])
    stream_gen = model.stream(input_ids, max_iter=max_iter, temperature=1.0, top_k=40)

    for output_ids in stream_gen:
        output_ids = output_ids.tolist()
        print(tokenizer.decode(output_ids[0]), end="", flush=True)
    return output_ids


if __name__ == "__main__":
    model = load_model()
    print("<prompt> My name is Teven and I am\n<generated> ")
    stream(model, "My name is Teven and I am", max_iter=100)
    print("\n====================\n")
    print("<prompt> I am a student at KAIST\n<generated> ")
    stream(model, "I am a student at KAIST", max_iter=100)
    print("\n====================\n")
    print("<prompt> I like to eat\n<generated> ")
    stream(model, "I like to eat", max_iter=100)