# Week2 by CL  

1. Iteration vs. Loss plots

> Modifiy train.py to run with RNN in RNNClassifier (nn.RNN or CL_RNN or CL_LSTM)

`$ python3 train.py`

`$ python3 train.py`

![RNN](RNN.png?raw=true "RNN")

![LSTM](LSTM.png?raw=true "LSTM")

  
2. Examples to show different results between LSTM and RNN

> Modifiy predict.py to run with saved model and vocab

`$ python3 predict.py` 
  
    tensor([[-0.1194, -0.2723, -0.5154, -0.4468]])
    This is a World news
    tensor([[-0.2225,  0.2112, -0.0264, -0.0376]])
    This is a Sports news
    tensor([[-0.1652, -0.7168, -0.5533,  0.2829]])
    This is a Sci/Tec news

In theory, some articles written in Sci/Tec with some Golf or Tennis examples may show different results between LSTM and RNN. 
LSTM may show more accurate result since this kind of article is better to be classified as Sci/Tec than Sports. 
