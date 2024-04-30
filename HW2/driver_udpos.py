import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

from Vocabulary import Vocabulary

import random
import torch
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils .data.backward_compatibility import worker_init_fn
import matplotlib.pyplot as plt
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn

from tqdm import tqdm

import pickle
 

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#POS
PARTS_OF_SPEECH = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

def get_part_of_speech_map():
    pos_map = {}

    for i, pos in enumerate(PARTS_OF_SPEECH):
        pos_encoding = np.zeros((len(PARTS_OF_SPEECH)))
        pos_encoding[i] = 1
        pos_map[pos] = pos_encoding
    
    return pos_map

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b [0] for b in batch]
    yy = [b [1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

# Visualizing POS tagged sentence
def visualizeSentenceWithTags(text, udtags):
    print(" Token "+"".join ([" "]*(15))+" POS Tag ")
    print(" ---------------------------------")
    for w, t in zip(text, udtags):
        print(w + "".join([" "]*(20 - len(w))) +t)

#Build vocab
def get_vocab(corpus):
    words = []
    for line in corpus:
        words += line[0]

    for i, w in enumerate(words):
        words[i] = w.lower()

    return Vocabulary(words)

#Load word vectors
def get_embedding_index():
    path_to_glove_file = "/home/magraz/AI539_NLP/HW2/word_vectors/glove.6B.50d.txt"

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    
    return embeddings_index

# Prepare embedding matrix
def get_embedding_matrix(vocab, embeddings_index):
    num_tokens = vocab.size + 2
    embedding_dim = 50
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in vocab.word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

class PartOfSpeechLSTM(torch.nn.Module) :

    def __init__(self, input_size=50, hidden_dim=100) :
        super().__init__()

        self.num_layers = 1
        self.input_dim = input_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim*2, 300)
        self.output = nn.Linear(300, 17)
      
        self.leaky = nn.LeakyReLU()
        self.softmax = nn.Softmax()

    def forward(self, x, s):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        out = self.leaky(out)
        out = self.output(out)
        out = self.softmax(out)
        return out


    def __str__(self):
        return "LSTM-"+str(self.hidden_dim)

def map_x_batch(vocab: Vocabulary, x_map, x_batch, max_len):

    for line in x_batch:

        diff = max_len - len(line)

        if diff == 0:
            continue
        
        for _ in range(diff):
            line.append('UNK')

    x_batch_vectors = []

    for words in x_batch:
        x_batch_vectors.append(list(map(lambda w: x_map[vocab.word2idx.get(w.lower(), vocab.word2idx['UNK'])], words)))

    return x_batch_vectors

def map_y_batch(y_map, y_batch, max_len):

    for line in y_batch:

        diff = max_len - len(line)

        if diff == 0:
            continue
        
        for _ in range(diff):
            line.append('X')

    y_batch_vectors = []

    for pos_list in y_batch:
        y_batch_vectors.append(list(map(lambda pos: y_map[pos], pos_list)))

    return y_batch_vectors

def train_model(model, train_loader, x_map, y_map, vocab, epochs=2000, lr=1e-3):
    #Train loss
    train_loss = []

    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adam is a version of SGD with dynamic learning rates 
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)

    # Main training loop over the number of epochs
    for i in tqdm(range(epochs), desc="Training: "):
        
        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for j, (x, y, lens) in enumerate(train_loader):

            max_len = max(lens)

            x = map_x_batch(vocab, x_map, x, max_len)
            y = map_y_batch(y_map, y, max_len)

            x = np.array(x)
            y = np.array(y)

            # push them to the GPU if we are using one
            x = torch.from_numpy(x.astype(np.float32)).to(dev)
            y = torch.from_numpy(y.astype(np.float32)).to(dev)

            # predict the parity from our model
            y_pred = model(x, lens)
            
            # compute the loss with respect to the true labels
            loss = crit(y_pred, y)
            
            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            # pred = torch.max(y_pred, 1)[1]
            # correct += (pred == y).float().sum()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f" % (i, sum_loss/total))#, val_loss, val_acc))
            train_loss.append(sum_loss/total)

    
    #Store model
    logging.info('Saving model...')
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model.pth')
    
    #Store loss
    with open('train_loss.pickle', 'wb') as handle:
        pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    # Create data pipeline
    train_data = datasets.UDPOS(split ='train')

    # Make data loader
    train_loader = DataLoader(
        dataset = train_data, batch_size =5,
        shuffle = True, num_workers =1 ,
        worker_init_fn = worker_init_fn ,
        drop_last = True, collate_fn = pad_collate)

    vocab = get_vocab(train_data)
    emb_idx = get_embedding_index()
    emb_mat = get_embedding_matrix(vocab, emb_idx)
    pos_map = get_part_of_speech_map()

    # Build the model and put it on the GPU
    logging.info("Building model")
    model = PartOfSpeechLSTM()
    model.to(dev) # move to GPU if cuda is enabled

    train_model(model, train_loader, x_map=emb_mat, y_map=pos_map, vocab=vocab)

    # mapped_x = []
    # mapped_y = []
    # last_x = 0
    # last_y = 0
    # for j, (x, y, l) in enumerate(train_loader):
    #     mapped_x = map_x_batch(vocab, emb_mat, x, max(l))
    #     mapped_y = map_y_batch(pos_map, y, max(l))

    #     last_x = x[1]
    #     last_y = y[1]
    #     break

    # print(np.array(mapped_y).shape)
    # print(np.array(mapped_x).shape)

if __name__== "__main__":
    main()

    with open('train_loss.pickle', 'rb') as handle:
        train_loss = pickle.load(handle)

    print(train_loss)

