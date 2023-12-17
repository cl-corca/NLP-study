import os
import json
import numpy as np
import torch
from torch import nn, optim
from dataloader import get_dataloader_and_vocab
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from model import CL_LSTM, CL_RNN

# number of epoch in training
EPOCHS = 15
LR = 1e-3
EMBED_LEN = 50 
HIDDEN_DIM = 50 
N_LAYERS = 1
DIR_W = 'data/rnn' #TODO (cl): 'rnn' for rnn_classifier, 'lstm' for lstm_classifier 

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_len, hidden_dim, n_layers, n_classes):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
        #TODO (cl)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        #self.rnn = CL_LSTM(input_size=embed_len, hidden_size=hidden_dim)
        #self.rnn = CL_RNN(input_size=embed_len, hidden_size=hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input):
        embeddings = self.embedding_layer(input)
        output, hidden = self.rnn(embeddings, torch.randn(self.n_layers, len(input), self.hidden_dim))
        return self.linear(output[:,-1])
    
def train():
    if not os.path.exists(DIR_W):
        os.makedirs(DIR_W)

    train_loader, vocab, classes = get_dataloader_and_vocab(
        ds_type="train",
        vocab=None,
        classes=None
    )

    test_loader, _, _ = get_dataloader_and_vocab(
        ds_type="test",
        vocab=vocab,
        classes=classes
    )

    vocab_size = len(vocab)
    model = RNNClassifier(vocab_size=vocab_size, embed_len=EMBED_LEN, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #to save loss after training
    loss_dict = {'train': [], 'test': []}

    for epoch in range(EPOCHS):
        print(f"start epoch {epoch+1}/{EPOCHS}")

        #train epoch
        running_loss = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)
            loss = criterion(Y_preds, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        print("Train Loss : {:.3f}".format(torch.tensor(running_loss).mean()))
        epoch_loss = np.mean(running_loss)
        loss_dict["train"].append(epoch_loss)

        #test epoch
        with torch.no_grad():
            Y_shuffled, Y_preds, running_loss = [],[],[]
            for X, Y in test_loader:
                preds = model(X)
                loss = criterion(preds, Y)
                running_loss.append(loss.item())
                Y_shuffled.append(Y)
                Y_preds.append(preds.argmax(dim=-1))
            
            Y_shuffled = torch.cat(Y_shuffled)
            Y_preds = torch.cat(Y_preds)

            print("Valid Loss : {:.3f}".format(torch.tensor(running_loss).mean()))
            print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))
            epoch_loss = np.mean(running_loss)
            loss_dict["test"].append(epoch_loss)


    #save model CL: change directory name based on dataset WikiSet2: 2, WikiSet103: 103
    model_path = os.path.join(DIR_W, "model.pt")
    torch.save(model, model_path)

    #save loss CL: change directory name based on dataset WikiSet2: 2, WikiSet103: 103
    loss_path = os.path.join(DIR_W, "loss.json")
    with open(loss_path, "w") as fp:
        json.dump(loss_dict, fp)

    #save vocab
    vocab_path = os.path.join(DIR_W, "vocab.pt")
    torch.save(vocab, vocab_path)

if __name__ == '__main__':
    train()