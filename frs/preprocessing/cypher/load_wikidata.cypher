// add to neo4j.conf 
// dbms.unmanaged_extension_classes=n10s.endpoint=/rdf

// initiate neosemantics conf file
CALL n10s.graphconfig.init({handleVocabUris: "IGNORE"});

// load subclasses of food (Q2095)
WITH '
PREFIX sch: <http://schema.org/> 
CONSTRUCT {
  ?class  a sch:Class ;
            sch:SUBCLASS_OF ?superclass ;
            sch:name ?class_name ;
            sch:off_id ?class_id .
  ?superclass a sch:Class ;
                sch:name ?superclass_name ;
                sch:off_id ?superclass_id .
}
WHERE {?class wdt:P279* wd:Q2095 ;
              rdfs:label ?class_name ;
              wdt:P279 ?superclass .
       ?superclass rdfs:label ?superclass_name ;
       OPTIONAL {?class wdt:P5930 ?class_id .}
       OPTIONAL {?superclass wdt:P5930 ?superclass_id .}
       FILTER (lang(?class_name) = "en")
       FILTER (lang(?superclass_name) = "en") } ' AS sparql

CALL n10s.rdf.import.fetch(
"https://query.wikidata.org/sparql?query=" +   
apoc.text.urlencode(sparql),"JSON-LD", 
{ headerParams: { Accept: "application/ld+json"} , 
handleVocabUris: "IGNORE"}) 
YIELD terminationStatus, triplesLoaded, extraInfo
RETURN terminationStatus, triplesLoaded;

// remove resource label
MATCH(r:Class:Resource)
REMOVE r:Resource
RETURN COUNT(r);

// load instances of food (Q2095)
WITH '
PREFIX sch: <http://schema.org/> 
CONSTRUCT {
  ?class  a sch:Class ;
            sch:INSTANCE_OF ?superclass ;
            sch:name ?class_name ;
            sch:off_id ?class_id .
  ?superclass a sch:Class ;
                sch:name ?superclass_name ;
                sch:off_id ?superclass_id .
}
WHERE {?class wdt:P31* wd:Q2095 ;
              rdfs:label ?class_name ;
              wdt:P31 ?superclass .
       ?superclass rdfs:label ?superclass_name ;
       OPTIONAL {?class wdt:P5930 ?class_id .}
       OPTIONAL {?superclass wdt:P5930 ?superclass_id .}
       FILTER (lang(?class_name) = "en")
       FILTER (lang(?superclass_name) = "en") } ' AS sparql

CALL n10s.rdf.import.fetch(
"https://query.wikidata.org/sparql?query=" +   
apoc.text.urlencode(sparql),"JSON-LD", 
{ headerParams: { Accept: "application/ld+json"} , 
handleVocabUris: "IGNORE"}) 
YIELD terminationStatus, triplesLoaded, extraInfo
RETURN terminationStatus, triplesLoaded;

// remove resource label
MATCH(r:Class:Resource)
REMOVE r:Resource
RETURN COUNT(r);
