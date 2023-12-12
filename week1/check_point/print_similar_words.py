import numpy as np
import torch
import sys

from sklearn.manifold import TSNE

sys.path.append("../")

folder = "weights/Wikiset2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(f"../{folder}/model.pt", map_location=device)
vocab = torch.load(f"../{folder}/vocab.pt")

# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape


def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings_norm[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

def print_top_similar(input_word):
  for word, sim in get_top_similar(input_word).items():
    print("{}: {:.3f}".format(word, sim))

if __name__ == "__main__":
    print_top_similar(input_word=sys.argv[1])
