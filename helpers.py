import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from anthropic import AuthenticationError as AnthropicAuthError
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import requests
import openai
import hashlib
import re
import os

def extract_core_error(report):
    """
    Extract the core error message, removing variable elements like timestamps, 
    line numbers, and URIs that might change between attempts
    """
    # Remove common variable elements
    cleaned = re.sub(r'line \d+', 'line X', report)
    cleaned = re.sub(r'column \d+', 'column X', cleaned)
    cleaned = re.sub(r'n\d+', 'nX', cleaned)  # Remove blank node identifiers
    cleaned = re.sub(r'#\w+', '#X', cleaned)  # Remove URI fragments
    cleaned = re.sub(r'http://[^\s]+', 'http://X', cleaned)  # Remove full URIs
    cleaned = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', cleaned)  # Remove dates
    cleaned = re.sub(r'\d+\.\d+', 'NUM', cleaned)  # Remove decimal numbers
    
    # Extract key error patterns
    error_patterns = [
        r'Constraint Violation',
        r'MinCount',
        r'MaxCount', 
        r'DataType',
        r'NodeKind',
        r'Class',
        r'Bad syntax',
        r'Expected',
        r'objectList expected',
        r'Missing',
        r'Invalid'
    ]
    
    core_errors = []
    for pattern in error_patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        core_errors.extend(matches)
    
    return ' '.join(core_errors).lower()

def is_syntax_error(report):
    """
    More precise detection of actual Turtle syntax errors vs SHACL validation errors
    """
    syntax_indicators = [
        "Bad syntax at line",
        "objectList expected",
        "Expected one of",
        "Unexpected end of file",
        "Invalid escape sequence",
        "Malformed URI",
        "Invalid character"
    ]
    
    # Must have syntax indicator AND not be a SHACL constraint violation
    has_syntax_indicator = any(indicator in report for indicator in syntax_indicators)
    has_shacl_violation = "Constraint Violation" in report or "sh:" in report
    
    return has_syntax_indicator and not has_shacl_violation

def should_retry_correction(report, previous_core_errors, max_same_error=2):
    """
    Determine if we should retry correction based on error patterns
    """
    core_error = extract_core_error(report)
    
    # Count how many times we've seen this core error
    same_error_count = previous_core_errors.count(core_error)
    
    return same_error_count < max_same_error

def safe_execute(func, *args, **kwargs):
    """Safely execute a function and handle all exceptions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Log the error for debugging (optional)
        print(f"Error: {e}")  # This will only show in your console, not to users
        raise e  # Re-raise to be caught by your main try-catch
    
def basic_syntax_cleanup(rdf_text, shacl_text):
    """
    Perform basic syntax cleanup that LLMs commonly mess up
    """
    # Common syntax fixes
    fixes = [
        # Remove malformed characters like '^b'
        (r"'\^b'", "''"),
        (r'"\^b"', '""'),
        # Fix malformed URI fragments
        (r"ex:([a-zA-Z0-9_]+)'\^b'", r"ex:\1"),
        # Remove stray quotes and characters
        (r"'[\^]+[a-zA-Z]+'", ""),
        # Fix malformed numeric values
        (r'""[\^]+[a-zA-Z]+""', '""'),
        # Clean up comment syntax issues
        (r'# [^;\n]*\n\n([a-zA-Z]+:)', r'# Comment\n\n\1'),
    ]
    
    cleaned_rdf = rdf_text
    cleaned_shacl = shacl_text
    
    for pattern, replacement in fixes:
        cleaned_rdf = re.sub(pattern, replacement, cleaned_rdf)
        cleaned_shacl = re.sub(pattern, replacement, cleaned_shacl)
    
    return cleaned_rdf, cleaned_shacl

def validate_turtle_syntax(turtle_text):
    """
    Check if Turtle syntax is valid by trying to parse it
    """
    try:
        temp_graph = Graph()
        temp_graph.parse(data=turtle_text, format="turtle")
        return True, "Valid syntax"
    except Exception as e:
        return False, str(e)

def validate_api_key(provider, api_key, endpoint=None):
    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
        elif provider == "Anthropic":
            client = Anthropic(api_key=api_key)
            client.models.list()
        elif provider == "Ollama":
            # For Ollama, we can just check if the endpoint is reachable
            if not endpoint:
                raise ValueError("No endpoint provided for Ollama.")
            response = requests.get(f"{endpoint}/v1/models", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama responded with status code {response.status_code}")
        return True, ""
    except openai.AuthenticationError:
        return False, "❌ **Invalid OpenAI API Key**\n\nPlease verify your API key at https://platform.openai.com/account/api-keys"
    except AnthropicAuthError:
        return False, "❌ **Invalid Anthropic API Key**\n\nPlease verify your API key at https://console.anthropic.com/settings/keys"
    except requests.exceptions.RequestException as e:
        return False, f"❌ **Cannot Connect to Ollama**\n\nFailed to connect to Ollama at `{endpoint}`. Please make sure Ollama is running and accessible. Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error while validating API key: {str(e)}"
    

def visualize_rdf(rdf_text):
            try:
                g = Graph().parse(data=rdf_text, format="turtle")
                nx_graph = nx.DiGraph()

                for s, p, o in g:
                    nx_graph.add_edge(str(s), str(o), label=str(p))

                # Create a larger network with improved physics settings
                net = Network(height="900px", width="100%", directed=True, notebook=False)
                
                # Configure physics for better graph spacing
                net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
                
                # Increase node spacing
                net.repulsion(node_distance=300, central_gravity=0.01, spring_length=300, spring_strength=0.05, damping=0.09)
                
                # Add nodes with larger size
                # Inside your visualize_rdf function, update the node addition logic:
                for node in nx_graph.nodes:
                    # Extract shorter node labels for readability
                    short_label = node.split("/")[-1] if "/" in node else node
                    short_label = short_label.split("#")[-1] if "#" in short_label else short_label
                    
                    # Check if this is a blank node (starts with 'n' followed by numbers)
                    is_blank_node = bool(re.match(r'^n\d+$', short_label))
                    
                    # Use different styling for blank nodes
                    if is_blank_node:
                        node_color = "#E8E8E8"  # Light gray
                        node_size = 15  # Smaller size
                        label = ""  # Hide the label
                    else:
                        node_color = "#97C2FC"  # Default blue
                        node_size = 25  # Normal size
                        label = short_label
                    
                    net.add_node(node, label=label, size=node_size, 
                                color=node_color, font={'size': 16}, 
                                title=node)  # Title shows on hover

                # Add edges with better visibility
                for u, v, d in nx_graph.edges(data=True):
                    # Extract shorter edge labels
                    edge_label = d["label"].split("/")[-1] if "/" in d["label"] else d["label"]
                    edge_label = edge_label.split("#")[-1] if "#" in edge_label else edge_label
                    
                    net.add_edge(u, v, label=edge_label, font={'size': 12}, width=1.5, title=d["label"])

                # Set options for better visualization
                net.set_options("""
                const options = {
                    "physics": {
                        "enabled": true,
                        "stabilization": {
                            "iterations": 100,
                            "updateInterval": 10,
                            "fit": true
                        },
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "springLength": 250,
                            "springConstant": 0.04,
                            "damping": 0.09
                        }
                    },
                    "layout": {
                        "improvedLayout": true,
                        "hierarchical": {
                            "enabled": false
                        }
                    },
                    "interaction": {
                        "navigationButtons": true,
                        "keyboard": true,
                        "hover": true,
                        "multiselect": true,
                        "tooltipDelay": 100
                    }
                }
                """)

                html_content = net.generate_html()
                return html_content
            
            except Exception as e:
                st.error(f"Failed to parse RDF: {e}")
                return None