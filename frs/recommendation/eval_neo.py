"""
Evaluate the baseline neo4j recommendation system. Query for all ingredients in a product except the last one, and calculate the accuracy of the recommendations.
"""
from argparse import ArgumentParser
from classes import ManyToOneData as Dataset
import pandas as pd
from py2neo import Graph
from rec_neo import recommend
import time

def main():
    # define arguments and parse
    parser = ArgumentParser()
    parser.add_argument('products', help='path to product df')
    args = parser.parse_args()
    # define connection parameters and connect to database
    url= 'bolt://localhost:7687'
    user = 'neo4j'
    password = 'admin'
    G = Graph(url, auth=(user, password))
    # read and augment data
    products = pd.read_pickle(args.products)
    # shuffle dataset to get all combinations
    recipes = Dataset.shuffle(None, products['ingredients'])
    # variables to store metrics and speed
    accuracy = {'1': 0, '5': 0, '10': 0}
    elapsed = []
    speed = 0
    # testing loop
    print('Progress:')
    for i, ingredients in enumerate(recipes, 1):
        if i % 100 == 0: 
            elapsed = []
        # start timer
        start = time.time()
        # print progress
        print(
            f'{i}/{len(recipes)} - {accuracy} - {speed:.2f} tics/sec', end='\r')
        # get the recipe X and remaining ingredient Y
        x, y = ingredients[:-1], ingredients[-1]
        # get recommendations
        recs = recommend(G, x, limit=10).to_ndarray(dtype=str)
        if recs.size > 0:
            # measure accuracy
            if y in recs[:1]: accuracy['1'] += 1
            if y in recs[:5]: accuracy['5'] += 1
            if y in recs: accuracy['10'] += 1
        elapsed.append(time.time()-start)
        # calculate speed
        speed = 1/(sum(elapsed)/len(elapsed))

    # write average accuracies
    print('Writing data.', end='\r')
    datadir = args.products.rsplit('/', 2)
    outdir = \
        datadir[0] + '/results/neo/' + datadir[-1].split('.')[0] + '_accs.txt'
    with open(outdir, 'w') as outfile:
        for k in accuracy: 
            outfile.write(
                f'Accuracy @ {k}:\t{accuracy[k] / len(products)}\n'
                )
    print(f'Calculated accuracies for {len(products)} combinations.')

if __name__ == '__main__':
    main()