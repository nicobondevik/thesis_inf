:source cypher/create_nodes.cypher;
RETURN 'Loaded nodes';
:source cypher/load_wikidata.cypher;
RETURN 'Loaded classes';
:source cypher/merge_nodes.cypher;
RETURN 'Merged nodes and classes';
:source cypher/load_rels.cypher;
RETURN 'Loaded relationships';
:source cypher/cleanup.cypher;
RETURN 'Properties cleaned';
:source cypher/co_occurrence.cypher
RETURN 'Co-occurrence relationships created';