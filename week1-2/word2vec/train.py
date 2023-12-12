import os
import json 
import numpy as np
import torch
from torch import nn, optim 
from torch.utils.data import DataLoader

from dataloader import get_dataloader_and_vocab
from constants import (
    EPOCHS, 
    EMBED_DIMENSION,
    EMBED_MAX_NORM,
    DIR_W,
) 

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

def train():
    os.makedirs(DIR_W)

    train_dataloader, vocab = get_dataloader_and_vocab(
        ds_type="train",
        vocab=None,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        ds_type="valid",
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    model = CBOW_Model(vocab_size=vocab_size) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.025) #learning_rate: 0.025
    lr_lambda = lambda epoch: (EPOCHS - epoch) / EPOCHS
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #to save loss after training 
    loss_dict = {'train': [], 'val': []}
    
    for epoch in range(EPOCHS):
        print(f"start epoch {epoch}/{EPOCHS}")

        #train epoch 
        model.train()
        running_loss = []
        for i, batch_data in enumerate(train_dataloader, 1):
            #print(f"train epoch: {i}")
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss) 
        loss_dict["train"].append(epoch_loss)    

        #validate epoch 
        model.eval()
        running_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_dataloader, 1):
                inputs = batch_data[0].to(device)
                labels = batch_data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        loss_dict["val"].append(epoch_loss)    

        lr_scheduler.step()
    
    #save model CL: change directory name based on dataset WikiSet2: 2, WikiSet103: 103
    model_path = os.path.join(DIR_W, "model.pt")
    torch.save(model, model_path)

    #save loss CL: change directory name based on dataset WikiSet2: 2, WikiSet103: 103
    loss_path = os.path.join(DIR_W, "loss.json")
    with open(loss_path, "w") as fp:
        json.dump(loss_dict, fp)    
 
if __name__ == '__main__':
    train()
