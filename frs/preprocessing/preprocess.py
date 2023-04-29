'''
This script prepares MongoDB JSON output for import into neo4j. 

First, ingredients occurring less than a set threshold are removed. Wikidata URIs are retrieved from ingredients.json where available for matching with URI in neo4j. If not, ingredient names are cleaned for matching with name in neo4j. Products, ingredients and their relationships are written to three CSV-files for import into Neo4j.

Usage:
    python3 neoprep.py [-c count_file] [-o output_directory] JSON-path

Author: Jon Nicolas Bondevik
'''

import argparse # parse command line arguments
import json     # parse JSON-file
import re       # clean product names
from collections import Counter # counter object for ingredient-counts
import nltk # extract nouns
from nltk.stem import WordNetLemmatizer # for lemmatization
# count ingredients and write to CSV-file
from ingredient_counter import count_ingredients, write_counts

def import_counts(path: str):
    '''
    Read counts from CSV-file.
    
    Parameters:
        path[str]: Path to CSV containing counts.
        
    Returns:
        counts[dict]: Dictionary containing instance as key and count as value
    '''
    counts = {}
    with open(path) as infile:
        for line in infile:
            id, count = line.rsplit(',', 1)
            counts[id] = int(count)
        return counts

def get_ingredients(counts: dict, n: int):
    '''
    Get ingredients occuring more n or more times.
    
    Parameters:
        counts[dict]: Dictionary with instance as key and count as value.
        n[int]: Count threshold for including instance.
        
    Returns:
        ingredients[set]: Set of instances with an occurrence above n.
    '''
    # filter out ingredients based on occurrence and language
    ingredients = {
        ing for ing, count in counts.items() if count  >= n and ing.startswith('en:')}
    return ingredients

def read_wiki(path):
    '''Read JSON containing ingredient ID and wikidata URI.
    
    Parameters:
        path[str]: Path to JSON containing ingredients.
        
    Returns:
        wiki[dict]: Dictionary with ID as key and URI as value.
    '''
    # read ingredients with wikidata URIs
    with open(path) as infile:
        wiki = dict() # dictionary to store all ingredients with wikidata URI
        for line in infile:
            line = json.loads(line.lower())
            for ingredient in line['tags']:
                uri = re.search(r"q\d+", ingredient.get('sameas', [''])[0])
                if uri:
                    ingredient_id = ingredient['id'].split(':', 1)[-1]
                    wiki[ingredient_id] = uri.group().upper()
        return wiki

def count_lines(path):
    '''Count number of lines in a file.'''
    with open(path) as infile:
        return sum(1 for line in infile)

def extract_nouns(string):
    '''Extracts lemmatized nouns from a string.'''
    lemmatizer = WordNetLemmatizer() # for lemmatization of nouns
    clean = list() # to store cleaned ingredients
    stopwords = {
        'lowfat', 'contains', 's', 'minute', 'invert', 'in'
    }
    # split on 'and'
    for ingredient in string.split(' and '):
        # tokenize string
        tokens = nltk.word_tokenize(ingredient)
        # remove stopwords (add all english stopwords here)
        tokens = [token for token in tokens if not token in stopwords]
        tags = nltk.pos_tag(tokens) # tag word classes
        nouns = [
            lemmatizer.lemmatize(word) for word, wclass in tags if wclass.startswith('NN')
        ]
        nouns = [word for word in nouns if word not in stopwords]
        clean.append('-'.join(nouns))
    return clean

def parse_json(path: str, ingredients_keep: set, wiki: dict):
    '''
    Parse JSON line by line to get products, ingredients and relationships.
    
    Parameters:
        path[str]: Path to JSON-file with products and ingredients.
        keep_ingredients[set]: Set of the IDs to keep.
        wiki_ingredients[dict]: Dictionary of ingredients with wiki URI.
    
    Returns:
        products[dict]: Dictionary with product IDs as keys and product names as values.
        ingredients[dict]: Dictionary with ID as key and URI as value.
        relationships[set]: Set of tuples in the form of product ID, ingredient ID and amount.
    '''
    with open(path) as infile:
        # count number of lines to keep track of progress
        total = count_lines(path)
        # initialize variables to store products, ingredients and relationships.
        products = dict()
        ingredients = dict()
        relationships = set()

        # iterate through JSON-objects
        for i, line in enumerate(infile, 1):
            print(f'{i}/{total}', end='\r')
            product = json.loads(line.lower())
            # remove non alphabetical characters, keep whitespace
            product_name = re.sub(r'[^a-z\s]', '', product['product_name'])
            # remove leading or trailing whitespace
            product_name = re.sub(r'^\s+|\s+$', '', product_name)
            # replace newline characters with space
            product_name = re.sub(r'\n', ' ', product_name)
            product_id = str(product['_id'])
            # skip parsing if product id is empty or already parsed
            if not product_id or products.get(product_id):
                continue
            # add product to dictionary
            products[product_id] = product_name
            
            # iterate through ingredients of one product
            for ingredient in product['ingredients']:
                # if ingredient is above occurrence threshold and english
                if ingredient['id'] in ingredients_keep:
                    # remove language tag
                    ingredient_id = [ingredient['id'].split(':', 1)[-1]]
                    # clean ingredients without corresponding URI
                    if not wiki.get(ingredient_id[0]):
                        # remove letters and symbols
                        ingredient_id = re.sub(r'[^a-z\-]', '', ingredient_id[0])
                        # replace dash with space
                        ingredient_id = ingredient_id.replace('-', ' ')
                        # extract nouns
                        ingredient_id = extract_nouns(ingredient_id)
                    # add ingredient(s) to dictionary
                    for id in ingredient_id:
                        if len(id) > 1: 
                            ingredients[id] = wiki.get(id, '')

                            # write relationship per ingredient
                            amount = float(ingredient.get('percent_estimate', 0))
                            if amount > 0:
                                relationships.add(
                                    (product['_id'], id, amount)
                                    )
        return products, ingredients, relationships

def write_products(path, products):
    '''
    Writes product ID and name to a CSV-file
    
    Parameters:
        path[str]: Directory to write to.
        products[dict]: Dictionary containing product IDs as keys and names as values.
        
    Returns:
        None.
    '''
    with open(path+'/products.csv', 'w') as outfile:
        outfile.write('id,name\n')
        for id, name in products.items():
            outfile.write(f'{id},"{name}"\n')

def write_ingredients(path, ingredients):
    '''
    Writes ingredient ID and URI to a CSV-file
    
    Parameters:
        path[str]: Directory to write to.
        ingredients[dict]: Dictionary containing product IDs as keys and names as values.
        
    Returns:
        None.
    '''
    with open(path+'/ingredients.csv', 'w') as outfile:
        outfile.write('name,uri\n')
        for id, uri in ingredients.items():
            outfile.write(f'{id},"{uri}"\n')

def write_relationships(path, relationships):
    '''
    Writes two files; ingredient IDs and weighted relationship between product and ingredients.
    
    Parameters:
        path[str]: Directory to write to.
        relationships: Set of tuples in the form of product ID, ingredient ID and amount.
    Returns:
        None.
    '''
    with open(path+'/relationships.csv', 'w') as outfile:
        outfile.write('product_id,ingredient_id,amount\n')
        for product_id, ingredient_id, amount in relationships:
            outfile.write(f'{product_id},"{ingredient_id}",{amount}\n')

def main():
    # add arguments to parser
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to MongoDB JSON.')
    parser.add_argument('-c', '--count', help='Optional count-file for quicker processing.')
    parser.add_argument('-o', '--out', help='Directory for writing output. Defaults to current directory.')
    # parse arguments
    args = parser.parse_args()

    path = args.path # infile path
    outdir = args.out if args.out else '.' # outfile path

    # filter ingredients based on occurrence
    if args.count: # count-file is given
        counts = import_counts(args.count)
    else:
        print('Counting ingredients.')
        counts = count_ingredients(path)
        print('Writing counts to CSV-file.')
        write_counts(outdir, counts=counts)
    ingredients = get_ingredients(counts, n=5)

    # get wikidata URIs
    wiki = read_wiki('data/raw/ingredients.json')

    # parse products
    print('Parsing JSON.')
    products, ingredients, relationships = parse_json(path, ingredients, wiki)

    # write CSV-files
    print('Writing products.')
    write_products(outdir, products)
    print('Writing ingredients')
    write_ingredients(outdir, ingredients)
    print('Writing relationships.')
    write_relationships(outdir, relationships)

if __name__ == '__main__':
    main()