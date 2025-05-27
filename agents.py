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
        self.system_prompt = "You are a knowledge engineer in materials science. Convert the following test data into RDF and SHACL."

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
Critique this RDF and SHACL for a materials ontology:
RDF:
{rdf_code}

SHACL:
{shacl_code}
"""
        return call_llm(prompt, "You are a SHACL critique expert.", self.model_info)

class OntologyMapperAgent:
    def __init__(self, model_info):
        self.model_info = model_info

    def run(self, user_input):
        prompt = f"Suggest ontology mappings for the following material science data:\n{user_input}"
        return call_llm(prompt, "You're a material science ontology assistant.", self.model_info)
