import torch 
from torch import nn 
from torchtext.data.utils import get_tokenizer
from model import CL_LSTM, CL_RNN

MAX_WORDS = 25 


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_len, hidden_dim, n_layers, n_classes):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
        #TODO (cl)
        self.rnn = CL_LSTM(input_size=embed_len, hidden_size=hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, input):
        embeddings = self.embedding_layer(input)
        output, hidden = self.rnn(embeddings, torch.randn(self.n_layers, len(input), self.hidden_dim))
        return self.linear(output[:,-1])
    
model = torch.load("data/cl_rnn/model.pt")
vocab = torch.load("data/cl_rnn/vocab.pt")
tokenizer = get_tokenizer("basic_english")
text_pipeline = lambda x: vocab(tokenizer(x))
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

def predict(text):
    with torch.no_grad():
        text = [vocab(tokenizer(text))]
        text = [tokens+([0]* (MAX_WORDS-len(tokens))) if len(tokens)<MAX_WORDS else tokens[:MAX_WORDS] for tokens in text] 
        text = torch.tensor(text, dtype=torch.int32)
        output = model(text)
        print(output)
        return output.argmax(1).item()+1 
    
ex_text_str1 = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
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

ex_text_str2 = "Congress often heads home at the end of the year \
    with a long to-do list left undone. But as Washington begins \
    to empty out this week, Congress is set to leave a remarkably \
    long menu of business world items on the table."

ex_text_str3 = "The major stock indices rallied higher for their seventh straight week"


print("This is a %s news" % ag_news_label[predict(ex_text_str1)])
print("This is a %s news" % ag_news_label[predict(ex_text_str2)])
print("This is a %s news" % ag_news_label[predict(ex_text_str3)])

