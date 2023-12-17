# Week2 by CL  

1. Iteration vs. Loss plots

> Modifiy train.py to run with RNN in RNNClassifier (nn.RNN or CL_RNN or CL_LSTM)

`$ python3 train.py`

`$ python3 train.py`

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
