## Week3 by CL  

Implement transformer 

# disclaimer 

경택님께 감사드립니다. 

- generate tokenizer and be ready to train 
  * `$ python3 datatool.py`

- train
  * `$ nohup python3 train.py > week3.out &`

- plot: using week3.out, update plot.py
  * `$ python3 plot.py`
    
![image](https://github.com/cl-corca/NLP-study/assets/loss.png)

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


  



  
