import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils .data.backward_compatibility import worker_init_fn

import numpy as np

import random

from tqdm import tqdm

import pickle

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b [0] for b in batch]
    yy = [b [1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

def prepare_sequence_x(batch, to_ix):
    seq = []

    for s in batch:
        seq += s
    
    idxs = [to_ix.get(w, to_ix['UNK']) for w in seq]

    return torch.tensor(idxs, dtype=torch.long).to(dev)

def prepare_sequence_y(batch, to_ix):
    seq = []

    for s in batch:
        seq += s
    
    idxs = [to_ix.get(w, to_ix['X']) for w in seq]

    return torch.tensor(idxs, dtype=torch.long).to(dev)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(dev)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True).to(dev)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim*2, tagset_size).to(dev)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(1, len(sentence), -1))
        tag_space = self.fc(lstm_out.view(len(sentence), -1))
        # tag_scores = F.softmax(tag_space, dim=1)
        return tag_space

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, train_loader, val_loader, x_map, y_map, epochs=1000, lr=1e-3):
    
    # Define a cross entropy loss function
    crit = nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adam is a version of SGD with dynamic learning rates 
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-5)

    #Train loss and acc
    train_loss = []
    train_acc = []

    #Validation loss and acc
    val_loss = []
    val_acc = []

    #Early stop

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    # Main training loop over the number of epochs
    for i in tqdm(range(epochs), desc="Training: "):
        
        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for _, (x, y, _) in enumerate(train_loader):

            x = prepare_sequence_x(x, x_map)
            y = prepare_sequence_y(y, y_map)

            # predict the parity from our model
            y_pred = model(x)
            
            # compute the loss with respect to the true labels
            loss = crit(y_pred, y)
            
            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum().item()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        train_loss.append(sum_loss/total)
        train_acc.append(correct/total)

        #Validation loop
        with torch.no_grad():
            model.eval()
            sum_loss = 0.0
            total = 0
            correct = 0

            for _, (x, y, _) in enumerate(val_loader):

                x = prepare_sequence_x(x, x_map)
                y = prepare_sequence_y(y, y_map)

                y_pred = model(x)

                loss = crit(y_pred, y)

                pred = torch.max(y_pred, 1)[1]
                correct += (pred == y).float().sum().item()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            
            val_loss.append(sum_loss/total)
            val_acc.append(correct/total)

        if i % 5 == 0:

            logging.info("epoch %d train loss %.3f, train acc %.3f val loss %.3f, val acc %.3f" % (i, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]))#, val_loss, val_acc))

            #Store model
            logging.info('Saving model...')
            torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'model.pth')
            
            #Store loss
            loss_acc = {'train_loss': train_loss, 
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                        }
            pickle_loss_acc = {}

            try:
                with open('loss_acc.pickle', 'rb') as handle:

                    pickle_loss_acc = pickle.load(handle)
                    pickle_loss_acc['train_loss'] += train_loss
                    pickle_loss_acc['train_acc'] += train_acc
                    pickle_loss_acc['val_loss'] += val_loss
                    pickle_loss_acc['val_acc'] += val_acc

                    logging.info(f"Total Epochs: {len(pickle_loss_acc['train_loss'])}")

            except FileNotFoundError:
                logging.info('Writing new loss file...')
                with open('loss_acc.pickle', 'wb') as handle:
                    pickle.dump(loss_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('loss_acc.pickle', 'wb') as handle:
                pickle.dump(loss_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            #Stop early
            if early_stopper.early_stop(val_loss[-1]):
                break

def main():
    # Create data pipeline
    train_data = datasets.UDPOS(split ='train')
    val_data = datasets.UDPOS(split= 'valid')

    # Make data loaders
    train_loader = DataLoader(
        dataset = train_data, batch_size =5,
        shuffle = True, num_workers =1 ,
        worker_init_fn = worker_init_fn ,
        drop_last = True, collate_fn = pad_collate)
    
    val_loader = DataLoader(
        dataset = val_data, batch_size =5,
        shuffle = True, num_workers =1 ,
        worker_init_fn = worker_init_fn ,
        drop_last = True, collate_fn = pad_collate)
    
    #Set idx for words
    word_to_ix = {}
    for _, (x, _, _) in enumerate(train_loader):
        for sent in x:
            for word in sent:
                if word not in word_to_ix:  # word has not been assigned an index yet
                    word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
    
    word_to_ix['UNK'] = len(word_to_ix) 

    #Set idx for tags
    tag_to_ix = {}
    tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    for i, tag in enumerate(tags):
        tag_to_ix[tag] = i

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

    train_model(model, train_loader, val_loader, x_map=word_to_ix, y_map=tag_to_ix)

if __name__== "__main__":
    main()
