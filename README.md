### NPL-study

## Week5 by CL  
- [Scaling Law](week5/scaling.md)
- [FLAN](week5/zeroshotlearners.md)
- [InstructGPT](week5/humanfeedback.md)
  
## Week4 by CL  

Implement GPT-nano

# disclaimer 
경택님 감사합니다. 
메모리가 부족해서 batch size를 16으로 경택님보다도 더 줄였습니다. 
batch size가 작아서인지 학습시간이 어마어마하게 걸리고 있습니다. 3일정도 돌렸음에도 epoch 0를 다 돌지 못했습니다. 

- save tokenizer 
  * `$ python3 datatool.py`

- train: this will store models `data/model.vx.epoch_x.step_xxxxx` and `data/loss.vx.json`
  * Even the first epoch takes more than 4 days in `hogwarts`  
  * `$ nohup python3 train.py > week4.out &`

- loss plot: using loss.vx.json, update plot.py (학습이 현재 된 거까지만을 바탕으로 그림) 
  * `$ python3 plot.py`
    
  ![image](https://github.com/cl-corca/NLP-study/assets/149552255/223c0567-62a0-4f7c-873a-002852f67041)

- ppl (45와는 거리가 먼 상태) 
  * `$ python3 test.py`

  perplexity: 160.8789102933949
  
- generate 
  * `$ python3 generate.py`
        
  ```My name is Teven and I am
  <generated>
   very famous when i say "teases", i have always tried to use a box on my car since i don't have a good enough shelf to handle. i like to use the rear of my car for my job and my computer, yet again i do wish i had been able to do it myself. in the process i wanted a new car in a different way. the thing that i loved is getting the best of them (especially if i'm getting this car) and
  
  Albert Einstein was a theoretical physicist
  <generated>
   at the time of the universe’s first dark matter and the first of many such experiments in history. by the end of the 19th century, his interest in creating the first black-light, the black-light and the dark was likely to be among the most popular experiments. he led the research of a class of particle physics that became known in the united states as the first black-light, the first black-light, first black-light, first black light. when his
  
  The Mona Lisa is a famous portrait painting
  <generated>
   based on the iconic character of david. her name has appeared in an online magazine and is an early addition to the brand. her paintings are in their current form, and their style remains the same. as the two sides of the paintings, the face and top of the paintings were on display for many years. they are known for their amazing work and she is currently living in calgary. she was one of thousands of artists working in the company. it is well known that she has
  ```
  

## Week3 by CL  

Implement transformer 

# disclaimer 

무엇보다도 경택님께 감사드립니다. 

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

- 3 examples: modify predict.py to run translate_examples()
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

![RNN](https://github.com/cl-corca/NLP-study/blob/main/week2/RNN.png?raw=true "RNN")

![LSTM](https://github.com/cl-corca/NLP-study/blob/main/week2/LSTM.png?raw=true "LSTM")

  
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




