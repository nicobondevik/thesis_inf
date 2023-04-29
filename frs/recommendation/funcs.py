"""All functions necessary for training and testing."""
from classes import Accumulator
import math
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.cuda import Device
from torchmetrics import Accuracy
import time

def grad_clipping(net, theta):
    """
    Clip gradients to avoid explosion.

    Parameters
    net : nn.Module
        A neural net.
    theta : int
        Strictness of gradient clipping.
    
    Returns
    None
    """
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_rnn(net, data_iter, loss, acc1, acc5, acc10, updater, device, 
                    use_random_iter):
    """
    Train Neural Net within one epoch.

    Parameters
    train_iter : DataLoader
        Training samples.
    loss : nn.Module
        Loss function.
    acc1 : Accuracy
        Object to measure accuracy@1.
    acc5 : Accuracy
        Object to measure accuracy@5.
    acc10 : Accuracy
        Object to measure accuracy@10.
    updater : nn.Module
        Optimizer for updating gradients.
    device : torch.device
        CPU or GPU
    use_random_iter : bool
        Samples randomly from batch if true.

    Returns
    metrics : tuple
        Loss, accuracy@1, accuracy@5 and accuracy@10.
    """
    state = None
    metric = Accumulator(4) # class to store loss and accuracies
    # start training mode
    net.train()
    # loop through batches one at a time (64 recipes)
    for X, Y in data_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # detach previous state from gpu memory
            state.detach_()
        # make sure dimensions of y will match y_hat
        Y = Y.reshape(-1)
        x, y = X.to(device), Y.to(device)
        # make prediction of the missing ingredient
        y_hat, state = net(x, state)
        # set gradients to zero (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
        updater.zero_grad()
        # calculate loss 
        l = loss(y_hat, y)
        # update gradients in backwards pass
        l.backward()
        # clip gradients (avoid exploding gradients)
        grad_clipping(net, 1)
        # update weights
        updater.step()
        # calculate accuracy
        acc = acc1(y_hat, y)
        acc_5 = acc5(y_hat, y)
        acc_10 = acc10(y_hat, y)
        metric.add(l, acc, acc_5, acc_10)
    # return average loss, and three accuracies
    metrics = math.exp(metric[0]/len(data_iter)), metric[1]/len(data_iter), \
        metric[2]/len(data_iter), metric[3]/len(data_iter)
    return metrics

def test_epoch_rnn(net, data_iter, loss, acc1, acc5, acc10, device):
    """
    Test Neural Net within one epoch.

    Parameters
    test_iter : DataLoader
        Unseen test samples.
    loss : nn.Module
        Loss function.
    acc1 : Accuracy
        Object to measure accuracy@1.
    acc5 : Accuracy
        Object to measure accuracy@5.
    acc10 : Accuracy
        Object to measure accuracy@10.
    device : torch.device
        CPU or GPU

    Returns
    metrics : tuple
        Loss, accuracy@1, accuracy@5 and accuracy@10.
    """
    state = None
    metric = Accumulator(4) # class to store loss and accuracies
    # set net to evaluation mode
    net.eval()
    # loop through recipes
    for X, Y in data_iter:
        if state is None:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # detach previous state from gpu memory
            state.detach_()
        # fix dimensions of y
        Y = Y.reshape(-1)
        # add to GPU
        x, y = X.to(device), Y.to(device)
        # make prediction
        y_hat, state = net(x, state)
        # calculate loss
        l = loss(y_hat, y)
        # get length without padding
        # calculate accuracy
        acc = acc1(y_hat, y)
        acc_5 = acc5(y_hat, y)
        acc_10 = acc10(y_hat, y)
        metric.add(l, acc, acc_5, acc_10)
    # return average loss, and three accuracies
    metrics = math.exp(metric[0]/len(data_iter)), metric[1]/len(data_iter), \
        metric[2]/len(data_iter), metric[3]/len(data_iter)
    return metrics

def train(net, train_iter, test_iter, lr, num_epochs, device, 
               use_random_iter=False):
    """
    Loop through all epochs to train and test a Recurrent Neural Net.

    Parameters
    net : nn.Module
        A Neural Net.
    train_iter : DataLoader
        Training samples.
    lr : int
        Learning rate.
    num_epochs : int
        Number of epochs.
    device : torch.device
        CPU or GPU
    use_random_iter : bool
        Samples randomly from batch if true.

    Returns
    train : dict
        Losses and accuracies for training set per epoch.
    test : dict
        Losses and accuracies for test set per epoch.
    """
    # add model to GPU
    net = net.to(device)

    # Initialize SGD optimizer, loss function and accuracy measures
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    acc = Accuracy(task='multiclass', num_classes=net.n_classes).to(device)
    acc5 = Accuracy(task='multiclass', num_classes=net.n_classes, 
                    top_k=5).to(device)
    acc10 = Accuracy(task='multiclass', num_classes=net.n_classes, 
                     top_k=10).to(device)

    # dicts to store perplexities and accuracies
    train = {'ppl': [], 'acc1': [], 'acc5': [], 'acc10': []}
    test = {'ppl': [], 'acc1': [], 'acc5': [], 'acc10': []}
    # list to store training times
    elapsed = []

    # train model and predict test set
    for epoch in range(num_epochs):
        # start timer
        start = time.time()
        # train model on training data and store metrics
        result = train_epoch_rnn(net, train_iter, loss, acc, acc5, acc10, 
                             optimizer, device, use_random_iter)
        train['ppl'].append(result[0])
        train['acc1'].append(result[1])
        train['acc5'].append(result[2])
        train['acc10'].append(result[3])

        # test model on unseen data and store metrics
        result = test_epoch_rnn(net, test_iter, loss, acc, acc5, acc10, device)
        test['ppl'].append(result[0])
        test['acc1'].append(result[1])
        test['acc5'].append(result[2])
        test['acc10'].append(result[3])

        # store elapsed time
        elapsed.append(time.time() - start)
        speed = sum(elapsed)/len(elapsed)
        # print current average speed
        print(f'{speed:.2f} seconds per epoch on {device}', end='\r')
    # print final average speed
    print(f'{speed:.2f} seconds per epoch on {device}')
    return train, test

def predict(prefix, preds, net, vocab, device):
    """
    Predict ingredients using trained neural net.

    Parameters:
    prefix : str
        Comma separated input ingredients.
    preds : int
        Number of ingredients to predict.
    net : nn.Module
        A neural net.
    vocab : Vocab
        Vocabulary of ingredients.
        
    Returns
    outputs : list
        A list of predictions.
    """
    state = net.begin_state(device, batch_size=1)
    prefix = prefix.split(',')
    inputs = [vocab.ing2idx[prefix[0]]]
    # reshape makes input for net 2D
    get_input = lambda: torch.tensor([inputs[-1]]).reshape((1, 1))
    for y in prefix[1:]:  # warm-up period for the hidden state
        _, state = net(get_input(), state)
        inputs.append(vocab.ing2idx[y])
    # predict final ingredient
    y, _ = net(get_input(), state)
    # get indices with highest probabilities
    y = y.sort(descending=True)[1].reshape(-1).tolist()
    # remove input ingredients from prediction
    for input in inputs:
        if input in y[:preds+1]: y.remove(input)
    # get most likely ingredients
    outputs = [vocab.ings[idx] for idx in y[:preds]]
    return outputs

def plot(train, test, path=None, dpi=300):
    """
    Plot results of training.

    Parameters:
    train : dict
        Losses and accuracies for training set per epoch.
    test : dict
        Losses and accuracies for test set per epoch.
    path : str
        Path to store plot.
    dpi : int
        Density Per Inch for plot.
    
    Returns
    None
    """
    # set labels for plots
    xlabel = 'Epoch'

    # plot train and test metrics
    fig, ax = plt.subplots(2,2, figsize=(20,15))
    # perplexity
    ax[0][0].plot(train['ppl'])
    ax[0][0].plot(test['ppl'])
    ax[0][0].set_xlabel(xlabel)
    ax[0][0].set_ylabel('Perplexity')
    ax[0][0].legend([f'Train ({train["ppl"][-1]:.2f})',
                     f'Test ({test["ppl"][-1]:.2f})'])

    # accuracy @ 1
    ax[0][1].plot(train['acc1'])
    ax[0][1].plot(test['acc1'])
    ax[0][1].set_xlabel(xlabel)
    ax[0][1].set_ylabel('Accuracy @ 1')
    ax[0][1].legend([f'Train accuracy @ 1 ({train["acc1"][-1]:.2f})',
                     f'Test accuracy @ 1 ({test["acc1"][-1]:.2f})'])    
    
    # accuracy @ 5
    ax[1][0].plot(train['acc5'])
    ax[1][0].plot(test['acc5'])
    ax[1][0].set_xlabel(xlabel)
    ax[1][0].set_ylabel('Accuracy @ 5')
    ax[1][0].legend([f'Train accuracy @ 5 ({train["acc5"][-1]:.2f})',
                     f'Test accuracy @ 5 ({test["acc5"][-1]:.2f})']
    )

    # accuracy @ 10
    ax[1][1].plot(train['acc10'])
    ax[1][1].plot(test['acc10'])
    ax[1][1].set_xlabel(xlabel)
    ax[1][1].set_ylabel('Accuracy @ 10')
    ax[1][1].legend([f'Train accuracy @ 10 ({train["acc10"][-1]:.2f})',
                     f'Test accuracy @ 10 ({test["acc10"][-1]:.2f})']
    )

    if path: plt.savefig(path, dpi=dpi)
    else: plt.show()
