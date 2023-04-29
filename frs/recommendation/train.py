"""
Trains the model.

Usage:
    python3 train.py data
"""
import argparse

import pandas as pd
import torch
from classes import RNN, RNNData, Vocab
from funcs import plot, train_rnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', help='Specify path to pickled dataframe with recipes.')
    parser.add_argument(
        '-o', '--out', help='Specify path to store model weights and plots.')
    args = parser.parse_args()

    # connect to available device and set manual seed for reproducability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)

    # define paths
    rec = args.data
    ing = args.data.rsplit('/', 1)[0] + '/ings.pkl'
    outdir = args.out + '_' if args.out else ''

    # read recipe data
    df = pd.read_pickle(rec)

    # subset data to 100,000 products
    df = df.head(100000)

    # build vocab
    vocab = Vocab(ing)

    # set batch size and length to consider
    batch_size, length = 1024, 20

    # split into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.25, random_state = seed)

    # read training data
    train_data = RNNData(train_df, vocab, length)
    train_iter = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    # read test data
    test_data = RNNData(test_df, vocab, length)
    test_iter = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)
    # read small subset for warming GPU
    warm_data = RNNData(train_df[:batch_size], vocab, length)
    warm_iter = DataLoader(warm_data, batch_size, shuffle=True, drop_last=True)

    # create model
    net = RNN(vocab, embed_dim=8, num_hiddens=128)

    # train and test model
    num_epochs, lr = 100, 1e-3
    print(f'Warming {device}')
    train, test = train_rnn(net, warm_iter, warm_iter, lr, 2, device)

    print(f'Training on {device}')
    train, test = train_rnn(net, train_iter, test_iter, lr, num_epochs, device)

    # save weights
    torch.save(net.state_dict(), f'{outdir}state_dict.pt')
    # plot results
    plot(train, test, f'{outdir}metrics.png')
    # write results
    with open(f'{outdir}metrics.txt', 'w') as out:
        out.write(
            f"""RESULTS
            --------------------------
            Batch size:          {batch_size}
            Recipe length:       {length}
            Number of epochs:    {num_epochs}
            Learning rate:       {lr}
            --------------------------
            Training samples:    {len(train_data)}
            Train perplexity:    {train['ppl'][-1]}
            Train accuracy:      {train['acc1'][-1]}
            Train accuracy @ 5:  {train['acc5'][-1]}
            Train accuracy @ 10: {train['acc10'][-1]} 
            --------------------------
            Testing samples:     {len(test_data)}
            Test perplexity:     {test['ppl'][-1]}
            Test accuracy:       {test['acc1'][-1]}
            Test accuracy @ 5:   {test['acc5'][-1]}
            Test accuracy @ 10:  {test['acc10'][-1]} 
            --------------------------
            """
        )
    print(f'Successfully saved model to {outdir}state_dict.pt')

if __name__ == '__main__':
    main()
