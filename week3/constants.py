import os
import torch

#TODO (cl): Set configurations properly  

#conf for train 
EPOCHS = 6
BETAS = (0.9, 0.98)
EPS = 1e-9
BATCH_SIZE = 250 
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_SEQUENCE_LENGTH = 50
WARMUP_STEPS = 8000
PAD_IDX = 1
VOCAB_SIZE = 47000

#device. change i of cuda:i to avoid busy GPU 
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

#conf for datatool
PATH_DATA = os.path.join(os.path.dirname(__file__), "data")
SPECIAL_SYMBOLS = ["<unk>", "<pad>", "<sos>", "<eos>", "<mask>"]
