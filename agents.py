from typing import Tuple
from utils import extract_rdf_shacl_improved, call_llm
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import re
import requests
from dotenv import load_dotenv
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