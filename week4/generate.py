import torch
from torch import IntTensor
from model import GPT
from datatool import get_tokenizer
from constants import *


def load_model():
    model = GPT()
    model.load_state_dict(torch.load("data/model.v2.epoch_0.step_95000"))
    model.to(DEVICE)
    model.eval()
    return model

def stream(model, prompt, max_iter = 100):
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = IntTensor([input_ids])
    stream_gen = model.stream(input_ids, max_iter=max_iter, temperature=1.0, top_k=40)

    for output_ids in stream_gen:
        output_ids = output_ids.tolist()
        print(tokenizer.decode(output_ids[0]), end="", flush=True)
    return output_ids


if __name__ == "__main__":
    model = load_model()
    prompt = "My name is Teven and I am"
    print(f"{prompt}\n<generated> ")
    stream(model, prompt)
    print("\n====================\n")

    prompt = "Albert Einstein was a theoretical physicist"
    print(f"{prompt}\n<generated> ")
    stream(model, prompt)
    print("\n====================\n")

    prompt = "The Mona Lisa is a famous portrait painting"
    print(f"{prompt}\n<generated> ")
    stream(model, prompt)
    print("\n====================\n")
