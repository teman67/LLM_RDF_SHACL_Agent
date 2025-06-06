from typing import Tuple
from utils import extract_rdf_shacl_improved, call_llm
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import re
import requests
from dotenv import load_dotenv
import os
import glob
from rdflib import Graph, Namespace, URIRef, Literal
from typing import Dict, List, Tuple, Set
import re
from difflib import SequenceMatcher
load_dotenv()

class RDFGeneratorAgent:
    def __init__(self, model_info):
        self.model_info = model_info
        self.system_prompt = """# Materials Science Knowledge Graph Expert

You are a specialized knowledge engineer for materials science, focusing on transforming unstructured creep test reports into standardized RDF and SHACL models. Your expertise bridges materials science domain knowledge with semantic web technologies.

## Core Competencies
- Converting materials testing data into formal ontology structures
- Creating valid, interoperable RDF representations of experimental data
- Generating SHACL shapes for validation and model conformance
- Maintaining knowledge graph best practices in materials science domains

## Structured Reasoning Approach

For each transformation task, follow this refined methodology:

### 1. Extract Entities and Concepts
- **Materials**: Composition, processing history, classification
- **Test Equipment**: Instruments, calibration status, standards compliance
- **Test Parameters**: Temperature, stress, atmosphere, loading protocols
- **Measurements**: Time series data, strain values, derived calculations
- **Personnel**: Operators, supervisors, analysts
- **Documentation**: Standards, procedures, certifications

### 2. Ontological Mapping
- Map each entity to appropriate ontology classes using:
  - Materials Workflow (`matwerk:`) - For material samples and testing procedures
  - Ontology for Biomedical Investigations (`obi:`) - For experimental processes
  - Information Artifact Ontology (`iao:`) - For documentation elements
  - NFDI/PMD Core (`nfdi:`, `pmd:`) - For domain-specific concepts
  - QUDT (`qudt:`, `unit:`) - For quantities, units, dimensions

### 3. Define Semantic Relationships
- Create object property networks reflecting physical and conceptual connections
- Establish provenance chains for data traceability using:
  - `obi:has_specified_input`/`obi:has_specified_output`
  - `prov:wasGeneratedBy`/`prov:wasDerivedFrom`
  - `matwerk:hasProperty`/`matwerk:hasFeature`
- Include bidirectional relationships with inverse properties

### 4. Quantitative Data Modeling
- Model all numerical values using the QUDT pattern:
  - Create quantity instances (e.g., `ex:temperature_value_001`)
  - Attach numerical values with `qudt:numericValue` and proper XSD types
  - Specify measurement units via `qudt:unit`
  - Include `qudt:standardUncertainty` where available

### 5. Temporal Data Representation
- Create observation collections with `time:Instant` timestamps
- Link time series data points to the relevant phase of creep behavior
- Maintain interval relationships for capturing test sequence

### 6. IRI Engineering & Metadata Enhancement
- Generate consistent, hierarchical IRIs following materials science conventions
- Add `rdfs:label` and `rdfs:comment` to ALL resources
- Include contextual metadata like creation date, version, and provenance

### 7. RDF Generation (Turtle Format)
- Create a complete, valid RDF document with comprehensive prefix declarations
- Group related triples for readability
- Include all required metadata and contextual information
- Follow W3C best practices for RDF representation

### 8. SHACL Shape Development
- Create node shapes for all major entity types
- Define property shapes with cardinality, value types, and constraints
- Include `sh:description` for human-readable validation messages
- Enforce required properties and data consistency rules

### 9. Validation & Refinement
- Test RDF against SHACL constraints
- Diagnose and resolve any validation issues
- Optimize for data quality and semantic correctness

## Required Namespace Declarations

Both RDF and SHACL outputs must include all of these prefixes:

```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/ns#> .
@prefix matwerk: <http://matwerk.org/ontology#> .
@prefix nfdi: <http://nfdi.org/ontology/core#> .
@prefix pmd: <http://pmd.org/ontology/core#> .
@prefix iao: <http://purl.obolibrary.org/obo/IAO_> .
@prefix obi: <http://purl.obolibrary.org/obo/OBI_> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
```

## Required Entity Types and Properties

### Core Material Entities
- `matwerk:MaterialSample` - Physical specimen undergoing testing
- `matwerk:Material` - Composition and classification of material
- `matwerk:MaterialProperty` - Properties like strength, ductility

### Experimental Process Entities
- `matwerk:CreepTest` - Main testing process
- `obi:assay` - General experimental process
- `obi:material_processing` - Sample preparation steps

### Information Entities
- `iao:document` - Test reports, procedures, standards
- `iao:measurement_datum` - Raw measurements
- `iao:data_set` - Collections of related measurements

### Measurement Entities
- `qudt:Quantity` - Quantitative values with units
- `time:Instant` - Temporal reference points
- `time:Interval` - Test duration periods

## Detailed Data Modeling Requirements

### 1. Sample Metadata Requirements
- Sample identification with traceable IRI pattern
- Material composition (elements and percentages)
- Processing history and heat treatment details
- Physical dimensions (gauge length, cross-section)
- Microstructural characteristics when available

### 2. Test Configuration Requirements
- Testing standard compliance (ASTM, ISO, etc.)
- Equipment details with calibration status
- Specimen geometry and orientation
- Environmental parameters (temperature, atmosphere)
- Loading conditions and control parameters

### 3. Measurement Requirements
- Time-strain data series with timestamps
- Creep rate calculations for each stage
- Rupture time or test termination point
- Derived properties (minimum creep rate, strain at rupture)
- Measurement uncertainties and confidence intervals

### 4. Results Representation Requirements
- Structured representation of primary, secondary, and tertiary creep phases
- Statistical summaries of key parameters
- Links to raw data and derived calculations
- Observations and analysis notes

## Output Deliverables

For each creep test report, generate two distinct artifacts:

1. **Complete RDF Data Model (Turtle format)**
   - Comprehensive representation of all extracted information
   - Properly typed entities with descriptive labels
   - Complete relationship network
   - Valid syntax with all required prefixes

2. **SHACL Validation Shape (Turtle format)**
   - Shape constraints matching exactly the RDF data structure
   - Property constraints with appropriate cardinality
   - Data type and value range enforcement
   - Validation reporting capabilities

Both outputs must be syntactically valid and semantically aligned with materials science domain knowledge. The SHACL shape must successfully validate the RDF data, producing a conformant validation report.

## Implementation Guidelines

- Use ontology design patterns from OBO Foundry when applicable
- Apply consistent naming conventions for all resources
- Include human-readable labels and descriptions for all entities
- Structure data hierarchically for navigation and query efficiency
- Ensure all numerical values have appropriate XSD datatypes
- Validate RDF and SHACL for syntax correctness before submission"""

    def run(self, user_input, previous_output=None):
        prompt = user_input
        if previous_output:
            prompt += f"\n\nImprove this RDF/SHACL:\n{previous_output}"

        llm_response = call_llm(prompt, self.system_prompt, self.model_info)
        return extract_rdf_shacl_improved(llm_response)

class ValidatorAgent:
    def run(self, rdf_code: str, shacl_code: str) -> Tuple[bool, str]:
        try:
            rdf_graph = Graph().parse(data=rdf_code, format="turtle")
            shacl_graph = Graph().parse(data=shacl_code, format="turtle")
            conforms, _, report = validate(data_graph=rdf_graph, shacl_graph=shacl_graph)
            return conforms, report
        except Exception as e:
            return False, f"Validation Error: {str(e)}"

class CritiqueAgent:
    def __init__(self, model_info):
        self.model_info = model_info

    def run(self, rdf_code, shacl_code):
        prompt = f"""
You are a senior materials science and knowledge graph engineer with expertise in semantic web technologies and ontology.

Please perform a detailed technical critique of the following RDF and SHACL output for a materials science knowledge graph and ontology:

RDF:
{rdf_code}

SHACL:
{shacl_code}

Provide a structured analysis covering:

1. SEMANTIC COHERENCE
   - Are domain concepts accurately represented?
   - Are relationships between entities semantically valid?
   - Is there proper use of materials science terminology?

2. STRUCTURAL INTEGRITY
   - Evaluate triple patterns and graph structure
   - Identify any disconnected nodes or subgraphs
   - Assess consistency in URI/IRI patterns

3. ONTOLOGY ALIGNMENT
   - Is the schema aligned with standard materials science ontologies (e.g., EMMO, MatOnto, ChEBI)?
   - Are there opportunities to link to established domain vocabularies?
   - Suggest specific namespace improvements

4. COMPLETENESS
   - Identify missing critical properties for materials characterization
   - Assess coverage of key materials relationships (composition, structure, properties, processing)
   - Evaluate sufficiency of metadata (provenance, units, measurement conditions)

5. SHACL VALIDATION
   - Are constraints appropriate for the domain?
   - Are validation rules comprehensive?
   - Are there missing constraints for ensuring data quality?

6. BEST PRACTICES
   - Conformance to Linked Open Data principles
   - Proper use of rdf:type, rdfs:subClassOf, owl:equivalentClass, etc.
   - Appropriate use of literals vs. URIs

Format your response with bullet points organized by these categories, and conclude with 2-3 highest priority recommendations for improvement.
"""
        return call_llm(prompt, "You are a SHACL critique expert.", self.model_info)

class OntologyMapperAgent:
    def __init__(self, model_info):
        self.model_info = model_info

    def run(self, user_input):
        prompt = f"Suggest ontology mappings for the following material science data:\n{user_input}"
        return call_llm(prompt, "You're a material science ontology assistant.", self.model_info)


class CorrectionAgent:
    def __init__(self, model_info):
        self.model_info = model_info

    def run(self, rdf_code, shacl_code, validation_report):
        retry_prompt = f"""
            Fix the SHACL validation errors in the following RDF and SHACL data.

            VALIDATION ERRORS:
            {validation_report}

            INSTRUCTIONS:
            1. Fix all validation errors listed above
            2. Return corrected RDF in first ```turtle block
            3. Return corrected SHACL in second ```turtle block
            4. Keep all namespace prefixes
            5. Ensure RDF conforms to SHACL shapes

            CURRENT RDF:
            ```turtle
            {rdf_code}
            ```

            CURRENT SHACL:
            ```turtle
            {shacl_code}
            ```

            Return the corrected versions now:
        """

        llm_response = call_llm(retry_prompt,"You are an expert at fixing SHACL validation errors in RDF data.",self.model_info)
        
        # return call_llm(retry_prompt, "You are an expert at fixing SHACL validation errors in RDF data.", self.model_info)
        return extract_rdf_shacl_improved(llm_response)
    

class OntologyMatcherAgent:
    def __init__(self, ontology_directory: str = "ontologies"):
        """
        Initialize the OntologyMatcherAgent
        
        Args:
            ontology_directory: Directory containing .ttl and .owl ontology files
        """
        self.ontology_directory = ontology_directory
        self.ontology_graphs = {}
        self.ontology_terms = {}
        self.load_ontologies()
    
    def load_ontologies(self):
        """Load all ontology files from the specified directory"""
        if not os.path.exists(self.ontology_directory):
            st.warning(f"Ontology directory '{self.ontology_directory}' not found. Creating directory...")
            os.makedirs(self.ontology_directory, exist_ok=True)
            return
        
        # Find all .ttl and .owl files
        ontology_files = []
        ontology_files.extend(glob.glob(os.path.join(self.ontology_directory, "*.ttl")))
        ontology_files.extend(glob.glob(os.path.join(self.ontology_directory, "*.owl")))
        
        if not ontology_files:
            st.info(f"No ontology files (.ttl or .owl) found in '{self.ontology_directory}' directory.")
            return
        
        with st.spinner("ontology files are Loading..."):
            for file_path in ontology_files:
                try:
                    filename = os.path.basename(file_path)
                    graph = Graph()
                    filenames = [os.path.basename(f) for f in ontology_files]
                    
                    # Determine format based on file extension
                    if file_path.endswith('.ttl'):
                        graph.parse(file_path, format="turtle")
                    elif file_path.endswith('.owl'):
                        graph.parse(file_path, format="xml")
                    
                    self.ontology_graphs[filename] = graph
                    self.ontology_terms[filename] = self._extract_terms(graph)
                    
                    # st.success(f"✅ Loaded ontology: {filename} ({len(self.ontology_terms[filename])} terms)")
                    
                except Exception as e:
                    st.error(f"❌ Failed to load {filename}: {str(e)}")
            
            st.success(f"✅ Ontology files: {', '.join(filenames)} are loaded successfully")
    
    def _extract_terms(self, graph: Graph) -> Dict[str, Dict]:
        """Extract terms (classes, properties, individuals) from an ontology graph"""
        terms = {}
        
        # Extract classes
        for subj in graph.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), 
                                  object=URIRef("http://www.w3.org/2002/07/owl#Class")):
            terms[str(subj)] = self._get_term_info(graph, subj, "Class")
        
        for subj in graph.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), 
                                  object=URIRef("http://www.w3.org/2000/01/rdf-schema#Class")):
            terms[str(subj)] = self._get_term_info(graph, subj, "Class")
        
        # Extract properties
        for subj in graph.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), 
                                  object=URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")):
            terms[str(subj)] = self._get_term_info(graph, subj, "ObjectProperty")
        
        for subj in graph.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), 
                                  object=URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty")):
            terms[str(subj)] = self._get_term_info(graph, subj, "DatatypeProperty")
        
        for subj in graph.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), 
                                  object=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property")):
            terms[str(subj)] = self._get_term_info(graph, subj, "Property")
        
        # Extract individuals/instances
        for subj in graph.subjects():
            if self._is_individual(graph, subj):
                terms[str(subj)] = self._get_term_info(graph, subj, "Individual")
        
        return terms
    
    def _get_term_info(self, graph: Graph, term: URIRef, term_type: str) -> Dict:
        """Get detailed information about a term"""
        info = {
            "type": term_type,
            "uri": str(term),
            "label": None,
            "comment": None,
            "local_name": self._extract_local_name(str(term))
        }
        
        # Get label
        for label in graph.objects(term, URIRef("http://www.w3.org/2000/01/rdf-schema#label")):
            info["label"] = str(label)
            break
        
        # Get comment/description
        for comment in graph.objects(term, URIRef("http://www.w3.org/2000/01/rdf-schema#comment")):
            info["comment"] = str(comment)
            break
        
        return info
    
    def _is_individual(self, graph: Graph, subj: URIRef) -> bool:
        """Check if a subject is an individual (not a class or property)"""
        # Skip if it's a class or property
        types_to_skip = [
            URIRef("http://www.w3.org/2002/07/owl#Class"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#Class"),
            URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
            URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"),
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property")
        ]
        
        for type_uri in types_to_skip:
            if (subj, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), type_uri) in graph:
                return False
        
        # Check if it has a type (indicating it's an individual)
        for _ in graph.objects(subj, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
            return True
        
        return False
    
    def _extract_local_name(self, uri: str) -> str:
        """Extract the local name from a URI"""
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        return uri
    
    def _extract_rdf_terms(self, rdf_code: str) -> Set[str]:
        """Extract terms used in the RDF code"""
        try:
            rdf_graph = Graph()
            rdf_graph.parse(data=rdf_code, format="turtle")
            
            terms = set()
            
            # Extract all URIs from subjects, predicates, and objects
            for subj, pred, obj in rdf_graph:
                if isinstance(subj, URIRef):
                    terms.add(str(subj))
                if isinstance(pred, URIRef):
                    terms.add(str(pred))
                if isinstance(obj, URIRef):
                    terms.add(str(obj))
            
            return terms
            
        except Exception as e:
            st.error(f"Error parsing RDF: {str(e)}")
            return set()
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity between two terms using multiple methods"""
        # Extract local names for comparison
        local1 = self._extract_local_name(term1).lower()
        local2 = self._extract_local_name(term2).lower()
        
        # Exact match
        if local1 == local2:
            return 1.0
        
        # Sequence similarity
        seq_sim = SequenceMatcher(None, local1, local2).ratio()
        
        # Check for substring matches
        if local1 in local2 or local2 in local1:
            substring_bonus = 0.3
        else:
            substring_bonus = 0.0
        
        # Check for common words (split by camelCase or underscores)
        words1 = set(re.findall(r'[A-Z][a-z]*|[a-z]+', local1))
        words2 = set(re.findall(r'[A-Z][a-z]*|[a-z]+', local2))
        
        if words1 and words2:
            word_intersection = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            word_sim = word_intersection / word_union if word_union > 0 else 0
        else:
            word_sim = 0
        
        # Combine similarities
        final_similarity = max(seq_sim, word_sim) + substring_bonus
        return min(final_similarity, 1.0)
    
    def find_matches(self, rdf_code: str, similarity_threshold: float = 0.7) -> Dict:
        """
        Find matches between RDF terms and ontology terms
        
        Args:
            rdf_code: The generated RDF code to analyze
            similarity_threshold: Minimum similarity score for matches (0.0 to 1.0)
            
        Returns:
            Dictionary containing match results
        """
        if not self.ontology_terms:
            return {
                "status": "no_ontologies",
                "message": "No ontology files loaded. Please add .ttl or .owl files to the ontologies/ directory.",
                "matches": []
            }
        
        rdf_terms = self._extract_rdf_terms(rdf_code)
        
        if not rdf_terms:
            return {
                "status": "no_rdf_terms",
                "message": "No terms extracted from RDF code.",
                "matches": []
            }
        
        all_matches = []
        exact_matches = []
        similar_matches = []
        
        for rdf_term in rdf_terms:
            for ontology_file, ontology_terms in self.ontology_terms.items():
                for onto_term_uri, onto_term_info in ontology_terms.items():
                    similarity = self._calculate_similarity(rdf_term, onto_term_uri)
                    
                    if similarity >= similarity_threshold:
                        match_info = {
                            "rdf_term": rdf_term,
                            "rdf_local_name": self._extract_local_name(rdf_term),
                            "ontology_file": ontology_file,
                            "ontology_term": onto_term_uri,
                            "ontology_local_name": onto_term_info["local_name"],
                            "ontology_type": onto_term_info["type"],
                            "ontology_label": onto_term_info["label"],
                            "ontology_comment": onto_term_info["comment"],
                            "similarity_score": similarity
                        }
                        
                        all_matches.append(match_info)
                        
                        if similarity == 1.0:
                            exact_matches.append(match_info)
                        else:
                            similar_matches.append(match_info)
        
        # Sort matches by similarity score (descending)
        all_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        similar_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "total_rdf_terms": len(rdf_terms),
            "total_ontology_terms": sum(len(terms) for terms in self.ontology_terms.values()),
            "total_matches": len(all_matches),
            "exact_matches": len(exact_matches),
            "similar_matches": len(similar_matches),
            "matches": all_matches,
            "similarity_threshold": similarity_threshold
        }
    
    def run(self, rdf_code: str, similarity_threshold: float = 0.7) -> str:
        """
        Main method to run the ontology matching analysis
        
        Args:
            rdf_code: The generated RDF code to analyze
            similarity_threshold: Minimum similarity score for matches
            
        Returns:
            Formatted string report of the analysis
        """
        results = self.find_matches(rdf_code, similarity_threshold)
        
        if results["status"] == "no_ontologies":
            return results["message"]
        
        if results["status"] == "no_rdf_terms":
            return results["message"]
        
        # Format the results as a readable report
        report = []
        report.append("# 🔍 Ontology Matching Analysis Report")
        report.append("")
        report.append(f"**Analysis Summary:**")
        report.append(f"- RDF Terms Analyzed: {results['total_rdf_terms']}")
        report.append(f"- Ontology Terms Available: {results['total_ontology_terms']}")
        report.append(f"- Total Matches Found: {results['total_matches']}")
        report.append(f"- Exact Matches: {results['exact_matches']}")
        report.append(f"- Similar Matches: {results['similar_matches']}")
        report.append(f"- Similarity Threshold: {results['similarity_threshold']}")
        report.append("")
        
        if results["matches"]:
            report.append("## 📋 Detailed Match Results")
            report.append("")
            
            current_rdf_term = ""
            for match in results["matches"]:
                if match["rdf_term"] != current_rdf_term:
                    current_rdf_term = match["rdf_term"]
                    report.append(f"### 🎯 RDF Term: `{match['rdf_local_name']}`")
                    report.append(f"*Full URI: {match['rdf_term']}*")
                    report.append("")
                
                match_type = "🎯 **EXACT MATCH**" if match["similarity_score"] == 1.0 else f"🔍 **SIMILAR MATCH** ({match['similarity_score']:.2f})"
                
                report.append(f"**{match_type}**")
                report.append(f"- **Ontology File:** {match['ontology_file']}")
                report.append(f"- **Ontology Term:** `{match['ontology_local_name']}`")
                report.append(f"- **Term Type:** {match['ontology_type']}")
                
                if match["ontology_label"]:
                    report.append(f"- **Label:** {match['ontology_label']}")
                
                if match["ontology_comment"]:
                    report.append(f"- **Description:** {match['ontology_comment'][:200]}{'...' if len(match['ontology_comment']) > 200 else ''}")
                
                report.append(f"- **Full URI:** {match['ontology_term']}")
                report.append("")
        else:
            report.append("## ❌ No Matches Found")
            report.append("")
            report.append("No terms in the generated RDF matched any terms in the loaded ontologies.")
            report.append(f"Consider lowering the similarity threshold (currently {results['similarity_threshold']}) or ")
            report.append("ensuring that relevant ontology files are placed in the ontologies/ directory.")
        
        return "\n".join(report)
    
    def replace_exact_matches(self, rdf_code: str, shacl_code: str, similarity_threshold: float = 0.7) -> Tuple[str, str, List[Dict]]:
        """
        Replace exact matches in RDF and SHACL with ontology terms
        
        Args:
            rdf_code: Original RDF code
            shacl_code: Original SHACL code  
            similarity_threshold: Minimum similarity for matches
            
        Returns:
            Tuple of (updated_rdf, updated_shacl, replacement_list)
        """
        # Get match results
        results = self.find_matches(rdf_code, similarity_threshold)
        
        if results["status"] != "success":
            return rdf_code, shacl_code, []
        
        # Filter for exact matches only
        exact_matches = [match for match in results["matches"] if match["similarity_score"] == 1.0]
        
        if not exact_matches:
            return rdf_code, shacl_code, []
        
        # Group matches by RDF term to avoid duplicate replacements
        unique_matches = {}
        for match in exact_matches:
            rdf_term = match["rdf_term"]
            if rdf_term not in unique_matches:
                unique_matches[rdf_term] = match
        
        replacement_list = []
        updated_rdf = rdf_code
        updated_shacl = shacl_code
        
        # Perform replacements
        for rdf_term, match in unique_matches.items():
            ontology_term = match["ontology_term"]
            
            # Create replacement record
            replacement_record = {
                "original_term": rdf_term,
                "replacement_term": ontology_term,
                "ontology_file": match["ontology_file"],
                "term_type": match["ontology_type"],
                "label": match["ontology_label"],
                "comment": match["ontology_comment"]
            }
            
            # Replace in RDF and SHACL
            # Use word boundaries to avoid partial replacements, but be careful with URIs
            if rdf_term.startswith('http'):
                # For full URIs, use exact string replacement
                updated_rdf = updated_rdf.replace(f'<{rdf_term}>', f'<{ontology_term}>')
                updated_shacl = updated_shacl.replace(f'<{rdf_term}>', f'<{ontology_term}>')
            else:
                # For prefixed terms, use word boundary replacement
                pattern = r'\b' + re.escape(rdf_term) + r'\b'
                updated_rdf = re.sub(pattern, ontology_term, updated_rdf)
                updated_shacl = re.sub(pattern, ontology_term, updated_shacl)
            
            replacement_list.append(replacement_record)
        
        # Handle namespaces for new ontology terms
        if replacement_list:
            updated_rdf, updated_shacl = self._extract_and_update_namespaces(
                updated_rdf, updated_shacl, replacement_list
            )
        
        return updated_rdf, updated_shacl, replacement_list

    def generate_replacement_report(self, replacement_list: List[Dict]) -> str:
        """Generate a human-readable report of replacements made"""
        if not replacement_list:
            return "## 🔄 Term Replacement Report\n\nNo exact matches found for replacement."
        
        report = []
        report.append("## 🔄 Term Replacement Report")
        report.append("")
        report.append(f"**{len(replacement_list)} exact matches replaced:**")
        report.append("")
        
        for i, replacement in enumerate(replacement_list, 1):
            report.append(f"### {i}. Term Replacement")
            report.append(f"- **Original Term:** `{replacement['original_term']}`")
            report.append(f"- **Replaced With:** `{replacement['replacement_term']}`")
            report.append(f"- **Source Ontology:** {replacement['ontology_file']}")
            report.append(f"- **Term Type:** {replacement['term_type']}")
            
            if replacement['label']:
                report.append(f"- **Label:** {replacement['label']}")
            
            if replacement['comment']:
                comment_preview = replacement['comment'][:150] + "..." if len(replacement['comment']) > 150 else replacement['comment']
                report.append(f"- **Description:** {comment_preview}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _extract_and_update_namespaces(self, rdf_code: str, shacl_code: str, replacement_list: List[Dict]) -> Tuple[str, str]:
        """
        Extract namespaces from ontology terms and add them to RDF/SHACL if needed
        
        Args:
            rdf_code: Current RDF code
            shacl_code: Current SHACL code
            replacement_list: List of replacements made
            
        Returns:
            Updated RDF and SHACL with necessary namespace declarations
        """
        # Extract existing namespaces
        existing_namespaces = set()
        namespace_pattern = r'@prefix\s+(\w+):\s+<([^>]+)>\s*\.'
        
        for match in re.finditer(namespace_pattern, rdf_code):
            prefix, uri = match.groups()
            existing_namespaces.add((prefix, uri))
        
        for match in re.finditer(namespace_pattern, shacl_code):
            prefix, uri = match.groups()
            existing_namespaces.add((prefix, uri))
        
        # Extract namespaces needed for replacement terms
        needed_namespaces = set()
        for replacement in replacement_list:
            ontology_term = replacement['replacement_term']
            
            # Extract namespace from URI
            if '#' in ontology_term:
                namespace_uri = ontology_term.rsplit('#', 1)[0] + '#'
            elif '/' in ontology_term:
                parts = ontology_term.rsplit('/', 1)
                namespace_uri = parts[0] + '/'
            else:
                continue
            
            # Create prefix from common ontology URIs or generate one
            prefix = self._generate_prefix_for_uri(namespace_uri)
            needed_namespaces.add((prefix, namespace_uri))
        
        # Add missing namespaces
        new_namespaces = needed_namespaces - existing_namespaces
        
        updated_rdf = rdf_code
        updated_shacl = shacl_code
        
        if new_namespaces:
            namespace_declarations = []
            for prefix, uri in new_namespaces:
                namespace_declarations.append(f"@prefix {prefix}: <{uri}> .")
            
            namespace_block = "\n".join(namespace_declarations) + "\n\n"
            
            # Add to beginning of files
            updated_rdf = namespace_block + updated_rdf
            updated_shacl = namespace_block + updated_shacl
        
        return updated_rdf, updated_shacl

    def _generate_prefix_for_uri(self, uri: str) -> str:
        """Generate a reasonable prefix for a namespace URI"""
        # Common ontology prefixes
        common_prefixes = {
            'http://www.w3.org/2002/07/owl#': 'owl',
            'http://www.w3.org/2000/01/rdf-schema#': 'rdfs',
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
            'http://www.w3.org/ns/shacl#': 'sh',
            'http://purl.obolibrary.org/obo/': 'obo',
            'http://xmlns.com/foaf/0.1/': 'foaf',
            'http://purl.org/dc/terms/': 'dcterms',
            'http://purl.org/dc/elements/1.1/': 'dc',
        }
        
        if uri in common_prefixes:
            return common_prefixes[uri]
        
        # Generate prefix from domain name or path
        try:
            if 'purl.org' in uri:
                return 'purl'
            elif 'w3.org' in uri:
                return 'w3'
            elif 'obolibrary.org' in uri:
                return 'obo'
            else:
                # Extract from domain or use generic
                import urllib.parse
                parsed = urllib.parse.urlparse(uri)
                domain_parts = parsed.netloc.split('.')
                if len(domain_parts) > 1:
                    return domain_parts[-2][:6]  # Use domain name, max 6 chars
                else:
                    return 'ns'  # Generic namespace
        except:
            return 'ns'
    
