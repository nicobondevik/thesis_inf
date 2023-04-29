// load product relationships
LOAD CSV WITH HEADERS FROM 'file:///data/relationships.csv' AS row
CALL {WITH row
MATCH (p:Product {barcode: row.product_id})
MATCH (i:Ingredient {og_name: row.ingredient_id})
MERGE (i)-[:INGREDIENT_IN {weight: row.amount}]->(p)
} IN TRANSACTIONS OF 10000 ROWS;