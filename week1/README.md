  # week1 
    
  1. Copied Olga's work and modified to make it work on my laptop 
    It is still running to make vetors for WikiText103. So tried to show check points 
    based on WikiText2
 
  2. print similar 10 words
     
  > at directory check_point
  
  `$ python3 print_similar_words.py happy` 
    
    ll: 0.685
    going: 0.615
    s: 0.599 
    t: 0.572
    ve: 0.570
    didn: 0.568
    don: 0.536
    girl: 0.519
    really: 0.516
    believe: 0.505

  `python3 print_similar_words.py tree`      
  
    indicating: 0.575
    theme: 0.556
    monument: 0.545
    image: 0.540
    planet: 0.538
    message: 0.527
    reference: 0.526
    painting: 0.519
    lies: 0.504
    sky: 0.503

 `python3 print_similar_words.py cloud`
 
    surface: 0.530
    color: 0.500
    outer: 0.488
    core: 0.478
    sound: 0.473
    atmosphere: 0.467
    eyes: 0.457
    flow: 0.456
    vocals: 0.455
    beyoncé: 0.442

  `python3 print_similar_words.py pencil`    
  
    Out of vocabulary word (학습되지 않은 단어 pencil) 

    
  3. calculate words
     
  > at directory check_point 
  
  `$ python3 cal_words king man woman` 
    
    king: 0.742
    jesus: 0.520
    earl: 0.515
    son: 0.510
    charles: 0.506
    reign: 0.504
    sir: 0.494
    republic: 0.490
    duke: 0.471
    part: 0.469

  `python3 cal_words.py bigger big small`
  
    small: 0.582
    <unk>: 0.513
    base: 0.444
    oil: 0.427
    garrison: 0.424
    predators: 0.422
    operations: 0.421
    aircraft: 0.418
    houses: 0.407
    fields: 0.401

 `python3 cal_words.py Paris France Germany`
 
    <unk>: 1.000
    words: 0.378
    scully: 0.370
    compounds: 0.368
    wings: 0.360
    genus: 0.351
    santa: 0.350
    cells: 0.346
    rome: 0.345
    murder: 0.345
    
-------------------- 
The Below is the original readme from Olga's github 
https://github.com/OlgaChernytska/word2vec-pytorch

# Word2Vec in PyTorch

Implementation of the first paper on word2vec - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). For detailed explanation of the code here, check my post - [Word2vec with PyTorch: Reproducing Original Paper](https://notrocketscience.blog/word2vec-with-pytorch-implementing-original-paper/).

## Word2Vec Overview

There 2 model architectures desctibed in the paper:

- Continuous Bag-of-Words Model (CBOW), that predicts word based on its context;
- Continuous Skip-gram Model (Skip-Gram), that predicts context for a word.

Difference with the original paper:

- Trained on [WikiText-2](https://pytorch.org/text/stable/datasets.html#wikitext-2) and [WikiText103](https://pytorch.org/text/stable/datasets.html#wikitext103) inxtead of Google News corpus.
- Context for both models is represented as 4 history and 4 future words.
- For CBOW model averaging for context word embeddings used instead of summation.
- For Skip-Gram model all context words are sampled with the same probability. 
- Plain Softmax was used instead of Hierarchical Softmax. No Huffman tree used either.
- Adam optimizer was used instead of Adagrad.
- Trained for 5 epochs.
- Regularization applied: embedding vector norms are restricted to 1.


### CBOW Model in Details
#### High-Level Model
![alt text](docs/cbow_overview.png)
#### Model Architecture
![alt text](docs/cbow_detailed.png)


### Skip-Gram Model in Details
#### High-Level Model
![alt text](docs/skipgram_overview.png)
#### Model Architecture
![alt text](docs/skipgram_detailed.png)


## Project Structure


```
.
├── README.md
├── config.yaml
├── notebooks
│   └── Inference.ipynb
├── requirements.txt
├── train.py
├── utils
│   ├── constants.py
│   ├── dataloader.py
│   ├── helper.py
│   ├── model.py
│   └── trainer.py
└── weights
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation

- **train.py** - script for training
- **config.yaml** - file with training parameters
- **weights/** - folder where expriments artifacts are stored
- **notebooks/Inference.ipynb** - demo of how embeddings are used

## Usage


```
python3 train.py --config config.yaml
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram", "cbow")
- dataset ("WikiText2", "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")


