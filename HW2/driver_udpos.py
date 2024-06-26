import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils .data.backward_compatibility import worker_init_fn

import nltk
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import random
import string
import numpy as np
from tqdm import tqdm
import pickle

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Get lemmatizer
nltk.download('wordnet')
nltk.download('punkt')
    
lemmatizer = WordNetLemmatizer()

#Set idx for tags
tag_to_idx = {}
idx_to_tag = {}
tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
for i, tag in enumerate(tags):
    tag_to_idx[tag] = i
    idx_to_tag[i] = tag

#Get pre trained word vectors into embedding map
path_to_glove_file = "glove.6B.50d.txt"

embeddings_map = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_map[word] = coefs

emb_dim = 50
hidden_dim = 50

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b [0] for b in batch]
    yy = [b [1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

def map_seq_x(batch, to_ix):
    seq = []

    for s in batch:
        seq += s
    
    idxs = np.array([to_ix.get(lemmatizer.lemmatize(w.lower()), to_ix['unk']) for w in seq])

    return torch.from_numpy(idxs).to(dev)

def map_seq_y(batch, to_ix):
    seq = []

    for s in batch:
        seq += s
    
    idxs = [to_ix.get(w, to_ix['X']) for w in seq]

    return torch.tensor(idxs, dtype=torch.long).to(dev)

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True).to(dev)

        self.fc = nn.Linear(hidden_dim*2, tagset_size).to(dev)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(1, len(sentence), -1))
        out = self.fc(lstm_out.view(len(sentence), -1))
        return out

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

def train_model(model, train_loader, val_loader, x_map, y_map, epochs=31, lr=1e-3):
    
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

    early_stopper = EarlyStopper(patience=2, min_delta=.10)

    # Main training loop over the number of epochs
    for i in tqdm(range(epochs), desc="Training: "):
        
        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for _, (x, y, _) in enumerate(train_loader):

            x = map_seq_x(x, x_map)
            y = map_seq_y(y, y_map)

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

                x = map_seq_x(x, x_map)
                y = map_seq_y(y, y_map)

                y_pred = model(x)

                loss = crit(y_pred, y)

                pred = torch.max(y_pred, 1)[1]
                correct += (pred == y).float().sum().item()
                sum_loss += loss.item()*y.shape[0]
                total += y.shape[0]
            
            val_loss.append(sum_loss/total)
            val_acc.append(correct/total)

        if i % 2 == 0:

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
                logging.info(f'Stopped early at {i} epochs')
                break

def train():
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

    model = BiLSTM(emb_dim, hidden_dim, len(tag_to_idx))

    train_model(model, train_loader, val_loader, x_map=embeddings_map, y_map=tag_to_idx)

def plot_train_val_loss():
    with open('loss_acc.pickle', 'rb') as handle:
        pickle_loss_acc = pickle.load(handle)
    
    epochs = range(1, len(pickle_loss_acc['train_loss']) + 1)

    plt.plot(epochs, pickle_loss_acc['train_loss'], label='Training loss')
    plt.plot(epochs, pickle_loss_acc['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def tag_sentence(sentence, model):

    processed_sent = []
    for w in sentence.split():
        
        if w[-1] in string.punctuation:
            w = w[:-1] + " " + w[-1]
        
        if w[0] in string.punctuation:
            w = w[0] + " " + w[1:]
    
        processed_sent.append(w)
        
    tokenized_sent = [w for w in " ".join(processed_sent).split()]

    with torch.no_grad():
        model.eval()

        x = map_seq_x([tokenized_sent], embeddings_map)

        y_pred = model(x)

        pred = torch.max(y_pred, 1)[1]

        print([idx_to_tag[idx] for idx in pred.tolist()])

def test_model(model):
    test_data = datasets.UDPOS(split= 'test')

    test_loader = DataLoader(
        dataset = test_data, batch_size =5,
        shuffle = True, num_workers =1 ,
        worker_init_fn = worker_init_fn ,
        drop_last = True, collate_fn = pad_collate)

    #Train loss and acc
    test_acc = []

    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0

        for _, (x, y, _) in enumerate(test_loader):

            x = map_seq_x(x, embeddings_map)
            y = map_seq_y(y, tag_to_idx)

            y_pred = model(x)

            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum().item()
            total += x.shape[0]

            test_acc.append(correct/total)
    
    #Plot test accuracy
    words = range(1, len(test_acc) + 1)

    plt.plot(words, test_acc, 'b', label='Validation accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Words')
    plt.ylabel('Accuracy')

    plt.show()

# Visualizing POS tagged sentence
def visualizeSentenceWithTags(text, udtags):
    print (" Token "+"". join ([" " ]*(15) )+" POS Tag ")
    print (" ---------------------------------")
    for w , t in zip ( text , udtags ):
        print (w+"".join([" "]*(20-len(w)))+t)

if __name__== "__main__":
    #Train
    #train()
    plot_train_val_loss()

    #Test
    model = BiLSTM(emb_dim, hidden_dim, len(tag_to_idx))
    model.load_state_dict(torch.load('model.pth', map_location=dev)['model_state_dict'])

    test_model(model)
    
    # Tag sentences
    sent = 'The old man the boat.'
    print(sent)
    tag_sentence(sent, model)
    print('')

    sent = 'The complex houses married and single soldiers and their families.'
    print(sent)
    tag_sentence(sent, model)
    print('')

    sent = 'The man who hunts ducks out on weekends.'
    print(sent)
    tag_sentence(sent, model)
    print('')
