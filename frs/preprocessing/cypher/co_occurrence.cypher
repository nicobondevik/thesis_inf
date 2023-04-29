CALL apoc.periodic.iterate(
'MATCH (i1:Ingredient)-[:INGREDIENT_IN]->(p:Product)<-[:INGREDIENT_IN]-(i2:Ingredient)
WHERE i1 <> i2
RETURN i1, i2, p',
'MERGE(i1)-[r:CO_PRODUCT]-(i2)
SET r.amount = COUNT(p)', {batchSize: 1000}) YIELD batches, total, operations;

