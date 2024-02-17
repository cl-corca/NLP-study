import os
import torch
from torch import nn, optim
from model import Transformer
from torch.optim.lr_scheduler import LRScheduler
from datatool import get_dataloader
from constants import *
   
class TransformerScheduler(LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    # To avoid, NotImplementedError, custom get_lr must be implemented 
    def get_lr(self):
        if self.last_epoch != 0:
            return [
                D_MODEL ** (-0.5) * min (self.last_epoch ** (-0.5), self.last_epoch * self.warmup_steps ** (-1.5)) 
                for _ in self.base_lrs
            ]
        return self.base_lrs

def train():
    # get_dataloader based on train.de and train.en in PATH_DIR 
    dataloader = get_dataloader()

    # set model, criterion, optimizer, lr_scheduler  
    model = Transformer(
        PAD_IDX, #TODO (cl): check PAD_IDX 
        VOCAB_SIZE, 
        D_MODEL, 
        NUM_HEADS, 
        NUM_LAYERS, 
        D_FF, 
        MAX_SEQUENCE_LENGTH, 
        DROPOUT
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), betas=BETAS, eps=EPS, lr=1e-7)
    lr_scheduler = TransformerScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    
    # train model and save it 
    model.to(DEVICE)
    loss_vec = []
    for epoch in range(EPOCHS):
        model.train()
        loss_per_epoch = 0.0 
        for step, (*data, label) in enumerate(dataloader):
            if data[0].device != DEVICE: 
                data = [d.to(DEVICE) for d in data]
                label = label.to(DEVICE)
            optimizer.zero_grad()
            pred = model(*data)
            pred = torch.transpose(pred, 1, 2)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            loss_per_epoch += batch_loss
            #print the progress every 1000 steps 
            if step % 1000 == 0:
                print(f"epoch: {epoch+1}/{EPOCHS} --> loss: {batch_loss}, step: {step}/{len(dataloader)}")
                step_model_path = os.path.join(f"data/transformer_model_{epoch}_{step}.json")
                torch.save(model.state_dict(), step_model_path)
            lr_scheduler.step()
        loss_per_epoch /= len(dataloader)
        loss_vec.append(loss_per_epoch)
    print(f"Final loss vector for graph: {loss_vec}")
    final_model_path = os.path.join("data/transformer_model_final.json")
    torch.save(model.state_dict(), final_model_path)
    print(f"Transfomer model is saved to {final_model_path}")

if __name__ == '__main__':
    train()
