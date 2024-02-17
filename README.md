### NPL-study

## Week3 by CL  

Implement transformer 

# disclaimer 

경택님께 감사드립니다. 

- generate tokenizer and be ready to train 
  * `$ python3 datatool.py`

- train: this will generate model `data/transformer_model_final.json`
  * 6 epochs and it takes about 15 hours in `hogwarts`  
  * `$ nohup python3 train.py > week3.out &`

- plot: using week3.out, update plot.py
  * `$ python3 plot.py`
    
![image](https://github.com/cl-corca/NLP-study/blob/main/week3/assets/loss.png?raw=true)

- bleu score: modify predict.py to run test()   
  * `$ nohup python3 predict.py > bleu.out`
  * 0.482 (48.2%) 

- 3 examples: modify predict.py to run generate_examples()
  *`$ python3 predict.py`

  >The quick brown fox jumps over the lazy dog.
  >
  >der schnelle bra une fox spring t über den f aul en hund .

  >I enjoy listening to classical music while I work.
  >
  >ich höre gerne klassische musik , während ich arbeite .

  >The sunset painted the sky in shades of pink and orange.
  >
  >der sonnenuntergang mal te den himmel in schatten von pink und orange .

# Week2 by CL  

1. Iteration vs. Loss plots

> Modifiy train.py to run with RNN in RNNClassifier (nn.RNN or CL_RNN or CL_LSTM)

`$ python3 train.py`

`$ python3 plot.py`

![RNN](RNN.png?raw=true "RNN")

![LSTM](LSTM.png?raw=true "LSTM")

  
2. Examples to show different results between LSTM and RNN

> Modifiy predict.py to run with saved model and vocab

  3 examples 
  
    "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

    "Congress often heads home at the end of the year \
    with a long to-do list left undone. But as Washington begins \
    to empty out this week, Congress is set to leave a remarkably \
    long menu of business world items on the table."

    "The major stock indices rallied higher for their seventh straight week"

2.1. Expected
`$ python3 predict.py` 
  
    This is a Sports news
    This is a Business news
    This is a Business news

2.2. CL_LSTM 

    tensor([[ 1.6159, -1.4454, -0.5996, -0.2585]])
    This is a World news
    tensor([[ 4.4231, -1.9324, -3.4669, -0.6937]])
    This is a World news
    tensor([[-0.9513, -4.3017,  4.8370, -0.7037]])
    This is a Business news

2.3. CL_RNN 

    tensor([[-0.2239, -0.1690, -0.4521, -0.2987]])
    This is a Sports news
    tensor([[-0.0540,  0.2565,  0.0030,  0.1729]])
    This is a Sports news
    tensor([[ 0.0847, -0.5083, -0.2501,  0.6198]])
    This is a Sci/Tec news


In theory, some articles written in Sci/Tec topic with Golf or Tennis examples later may show different results between LSTM and RNN. 
LSTM may show more accurate result as Sci/Tec while RNN show as Sports based only by examples. 
  
## Week1 
    
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



