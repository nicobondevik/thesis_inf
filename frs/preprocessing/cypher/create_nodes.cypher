CALL apoc.schema.assert({},{Ingredient:['og_name'], Product:['barcode'], Resource:['uri']});

// load ingredient nodes
LOAD CSV WITH HEADERS FROM 'file:///data/ingredients.csv' AS row
CALL {WITH row
MERGE (i:Ingredient {og_name: row.name, uri: row.uri})
} IN TRANSACTIONS OF 10000 ROWS;

// load product nodes
LOAD CSV WITH HEADERS FROM 'file:///data/products.csv' AS row
CALL {WITH row
MERGE (p:Product {barcode: row.id, name: row.name})
} IN TRANSACTIONS OF 10000 ROWS;
