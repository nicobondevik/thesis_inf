'''
Script to count the occurrence of ingredients throghout all products. The occurrence is written to a CSV-file that can be given to neoprep for faster execution.

Usage:
    python3 ingredient_counter.py JSON-file

Author: Jon Nicolas Bondevik
'''

import argparse # parse command line arguments
from collections import Counter # counter object for ingredient-counts
import json # parse JSON

def count_ingredients(path: str):
    '''
    Count the occurence of ingredients from MongoDB JSON.
    
    Parameters:
        path[str]: Path to JSON.
        
    Returns:
        counts[dict]: Dictionary-like counter object.
    '''
    with open(path) as infile:
        counts = Counter() # object for counting
        for line in infile: # iterate through products
            product = json.loads(line.lower()) 
            for ingredient in product['ingredients']: # iterate through ingredients
                counts.update([ingredient['id']]) # update count of ID
        return counts

def write_counts(path: str, counts: dict):
    '''
    Write dictionary with ingredients as keys and counts as values.
    
    Parameters:
        path[str]: Directory to write to.
        counts[dict]: Dictionary with ingredients as keys and counts as values.
    
    Returns:
        None
    '''
    with open(path+'/ingredient_counts.csv', 'w') as outfile:
        for id, count in counts.items():
            outfile.write(f'{id},{count}\n')

def main():
    # add arguments to parser
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to MongoDB JSON.')
    parser.add_argument('-o', '--out', help='Directory for writing output. Defaults to current directory.')
    # parse arguments
    args = parser.parse_args()

    path = args.path
    outdir = args.out if args.out else '.'
    
    print('Counting ingredients.')
    counts = count_ingredients(path=path)
    print('Writing ingredient-counts.')
    write_counts(outdir, counts=counts)

if __name__ == '__main__':
    main()