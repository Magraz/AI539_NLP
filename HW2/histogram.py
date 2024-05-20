import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import torch
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils .data.backward_compatibility import worker_init_fn
import matplotlib.pyplot as plt
import numpy as np

#Font sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# Create data pipeline
train_data = datasets.UDPOS(split='train')

# Function to combine data elements from a batch
def pad_collate(batch):
    xx = [b [0] for b in batch]
    yy = [b [1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

# Make data loader
train_loader = DataLoader(
    dataset = train_data, batch_size =5,
    shuffle = True, num_workers =1 ,
    worker_init_fn = worker_init_fn ,
    drop_last = True, collate_fn = pad_collate)

# Look at the first batch
xx ,yy, xlens = next(iter(train_loader))

# Visualizing POS tagged sentence
def visualizeSentenceWithTags(text, udtags):
    print(" Token "+"".join ([" "]*(15))+" POS Tag ")
    print(" ---------------------------------")
    for w, t in zip(text, udtags):
        print(w + "".join([" "]*(20 - len(w))) +t)

def plot_histogram(dataloader):

    resultList = []
    for _,yy, _ in iter(dataloader):
        resultList += sum(yy, [])

    labels, counts = np.unique(resultList,return_counts=True)

    print(len(resultList))
    print(counts[list(labels).index('NOUN')])
    print(f"Accuracy: {100*counts[list(labels).index('NOUN')]/len(resultList)}%")

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ticks = range(len(counts))
    plt.bar(ticks,counts, align='center')
    plt.xticks(ticks, labels)
    plt.title("Part-Of-Speech Histogram")

    plt.show()


# visualizeSentenceWithTags(xx[0], yy[0])
plot_histogram(train_loader)



