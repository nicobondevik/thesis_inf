// Remove original name of merged nodes
MATCH(i:Ingredient)
WHERE 'Class' IN LABELS(i)
REMOVE i.og_name
RETURN COUNT(i);

// Rename original name of unmerged nodes
MATCH(i:Ingredient)
WHERE NOT 'Class' IN LABELS(i) AND i.name IS NULL
SET i.name = i.og_name
REMOVE i.og_name
RETURN COUNT(i);