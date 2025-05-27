from pyshacl import validate

def validate_rdf_shacl(rdf_code, shacl_code):

    try:
        # Parse RDF with better error handling
        rdf_graph = Graph()
        rdf_graph.parse(data=rdf_code, format="turtle")
        
        # Parse SHACL with better error handling  
        shacl_graph = Graph()
        shacl_graph.parse(data=shacl_code, format="turtle")

        # Validate with more detailed reporting
        conforms, results_graph, results_text = validate(
            data_graph=rdf_graph,
            shacl_graph=shacl_graph,
            inference='rdfs',
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
            debug=False
        )
        
        # Return more detailed results
        if results_text:
            return conforms, results_text
        else:
            return conforms, "Validation completed successfully" if conforms else "Validation failed with unknown errors"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}\n\nThis might be due to invalid Turtle syntax. Please check the RDF and SHACL code for syntax errors."

