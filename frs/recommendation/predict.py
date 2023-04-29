"""
Script to use trained network weights to recommend ingredients.

Usage:
    python3 predict.py model_weights ingredients

Author:
    Jon Nicolas Bondevik
"""
from argparse import ArgumentParser
from funcs import predict
from classes import Vocab, RNN
import torch

def main():
    # find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # add and parse args
    parser = ArgumentParser()
    parser.add_argument('model_weights', help='Path to state dict containing model weights.')
    parser.add_argument('ingredients', help='Ingredients separated by commas.')
    parser.add_argument(
        '-n', default=10, type=int,help='Number of recommendations.')
    args = parser.parse_args()
    # create vocab
    vocab = Vocab('../data/dataframes/ings.pkl')
    # create model
    net = RNN(vocab, num_hiddens=128, embed_dim=8)
    # load weights
    net.load_state_dict(torch.load(args.net, map_location=device))
    # predict
    recs = predict(args.ingredients, args.n, net, vocab, device)
    for i, (rec) in enumerate(recs, 1):
        print(f'{i:>2}: {rec}')

if __name__ == '__main__':
    main()