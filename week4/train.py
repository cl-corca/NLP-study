import os

from datasets import load_dataset
from torch import nn

from scheduling_trainer import LRStepSchedulingTrainer
from constants import *
from model import GPT
from datatool import get_dataloader
        

def train():
    # get_dataloader based on train.de and train.en in PATH_DIR
    dataloader = get_dataloader()

    # set model, criterion, optimizer, lr_scheduler
    model = GPT()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = model.get_optimizer()
    trainer = LRStepSchedulingTrainer(
        model=model,
        train_dataloader=dataloader,
        batch_size=BATCH_SIZE,
        criterion=criterion,
        optimizer=optimizer,
    )
    trainer.run(
        num_epoch=EPOCH,
        device=DEVICE,
        model_save_path=os.path.join(PATH_DATA, "model"),
        loss_save_path=os.path.join(PATH_DATA, "loss"),
        model_version="v0",
        verbose=True,
    )

if __name__ == '__main__':
    train()