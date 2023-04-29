// MERGE NODES

// merge wikidata instances with wikidata classes
CALL apoc.periodic.iterate(
    'MATCH(c1:Class)
    MATCH(c2:Class {uri: c1.uri})
    WHERE id(c1) <> id(c2)
    RETURN id(c1) as c1_id, id(c2) as c2_id',
    'MATCH (c1:Class) WHERE id(c1) = c1_id
    MATCH (c2:Class) WHERE id(c2) = c2_id
    CALL apoc.refactor.mergeNodes([c1, c2], {
        properties:"discard",
        mergeRels:true
    }) YIELD node
    RETURN node', {batchSize: 1}) YIELD batches, total, operations;

// match on uri
CALL apoc.periodic.iterate(
    'MATCH (i:Ingredient)
    WHERE NOT "Class" IN labels(i) AND i.uri <> ""
    MATCH (c:Class) WHERE split(c.uri, "/")[-1] = i.uri
    RETURN id(i) AS i_id, id(c) AS c_id',
    'MATCH (i:Ingredient) WHERE id(i) = i_id
    MATCH (c:Class) WHERE id(c) = c_id
    CALL apoc.refactor.mergeNodes([c, i], {
        properties:"discard",
        mergeRels:true
    }) YIELD node
    RETURN node', {batchSize: 1}) YIELD batches, total, operations;

// match unmatched ingredients on open food facts id
CALL apoc.periodic.iterate(
    'MATCH (i:Ingredient)
    WHERE NOT "Class" IN labels(i)
    MATCH (c:Class) 
    WHERE c.off_id IS NOT NULL AND c.off_id = i.og_name
    RETURN id(i) AS i_id, id(c) AS c_id',
    'MATCH (i:Ingredient) WHERE id(i) = i_id
    MATCH (c:Class) WHERE id(c) = c_id
    CALL apoc.refactor.mergeNodes([c, i], {
        properties:"discard",
        mergeRels:true
    }) YIELD node
    RETURN node',
    {batchSize: 1}) YIELD batches, total, operations;

// match on cleaned ingredient name with 'en:'
CALL apoc.periodic.iterate(
    'MATCH (i:Ingredient)
    WHERE NOT "Class" IN labels(i)
    MATCH (c:Class)
    WHERE c.off_id IS NOT NULL AND c.off_id = "en:" + i.og_name
    RETURN id(i) AS i_id, id(c) AS c_id',
    'MATCH (i:Ingredient) WHERE id(i) = i_id
    MATCH (c:Class) WHERE id(c) = c_id
    CALL apoc.refactor.mergeNodes([c, i], {
        properties:"discard",
        mergeRels:true
    }) YIELD node
    RETURN node',
    {batchSize: 1}) YIELD batches, total, operations;
