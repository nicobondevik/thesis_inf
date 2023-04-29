from py2neo import Graph
import argparse
"""
Baseline recommendation system. Recommends ingredients co-occurring with all input ingredients. Recommendations are sorted by their co-occurence frequency.
"""
def recommend(graph, x, limit):
    """
    Queries cypher for co-occurring ingredients.
    Parameters
        graph: a py2neo graph object connected to neo4j
        x: input ingredients
        limit: number of recommendations to make
    Returns
        y: a cursor object for navigating recommendations
    """
    query = r'''
    // get input ingredients
    WITH $input AS ingredients // get input ingredients
    MATCH (i:Ingredient) WHERE i.name IN ingredients

    // match ingredients co-occurring with the input ingredients
    MATCH (i)-[r:CO_CLASS|CO_PRODUCT]-(rec:Ingredient)

    // match all co-occurrences of ingredient and get amount
    MATCH (i:Ingredient)-[r:CO_CLASS|CO_PRODUCT]-(rec)
    WITH ingredients, rec, COLLECT(i.name) AS matches, COLLECT(r.amount) AS amount

    // where co-occurring ingredient is not in input, and occurs with all inputs
    WHERE NOT rec.name IN ingredients AND ALL(ing IN ingredients WHERE ing IN matches)

    // sort by the co-occurrence of matched ingredient to all input ingredients
    WITH rec.name AS ingredient, apoc.coll.sum(amount) AS occurrence
    ORDER BY occurrence DESC
    RETURN ingredient LIMIT $limit
    '''
    y = graph.run(query, parameters ={'input': x, 'limit': limit})
    return y

def main():
    # define arguments and parse
    parser = argparse.ArgumentParser()
    parser.add_argument('ingredients', help='Input ingredients.')
    parser.add_argument('-l', '--limit', nargs='?', default=10, type=int)
    args = parser.parse_args()
    x = args.ingredients.split(', ')
    # define connection parameters and connect to database
    url= 'bolt://localhost:7687'
    user = 'neo4j'
    password = 'admin'
    G = Graph(url, auth=(user, password))
    # get recommendations
    recs = recommend(G, x, args.limit)
    # print recommendations
    for i, (rec, _) in enumerate(recs,1):
        print(f'{i:>2}: {rec}')

if __name__ == '__main__':
    main()