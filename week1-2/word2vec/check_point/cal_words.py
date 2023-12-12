import numpy as np
import torch
import sys
from torch import nn

from sklearn.manifold import TSNE

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

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



def cal_words(word1, word2, word3):
  emb1 = embeddings[vocab[word1]]
  emb2 = embeddings[vocab[word2]]
  emb3 = embeddings[vocab[word3]]

  emb4 = emb1 - emb2 + emb3
  emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
  emb4 = emb4 / emb4_norm

  emb4 = np.reshape(emb4, (len(emb4), 1))
  dists = np.matmul(embeddings_norm, emb4).flatten()

  top10 = np.argsort(-dists)[:10]

  for word_id in top10:
    print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))


if __name__ == "__main__":
    cal_words(word1=sys.argv[1], word2=sys.argv[2], word3=sys.argv[3])
