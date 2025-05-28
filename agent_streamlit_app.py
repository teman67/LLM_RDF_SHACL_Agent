import streamlit as st
from controller import SemanticPipelineAgent
from dotenv import load_dotenv
import os
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

load_dotenv()


st.set_page_config(page_title="SemAiOn Agent", layout="wide")
st.title("🧠 SemAiOn: Agent-Based Semantic Data Generator")

# Sidebar: API Configuration
st.sidebar.header("🔐 API Configuration")
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Ollama"])
if provider == "OpenAI":
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    model = st.sidebar.selectbox("OpenAI Model", model_options, index=0)
elif provider == "Anthropic":
    model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
    model = st.sidebar.selectbox("Claude Model", model_options, index=1)
else:
    model_options = ["llama3.3:70b-instruct-q8_0", "qwen3:32b-q8_0", "phi4-reasoning:14b-plus-fp16"]
    model = st.sidebar.selectbox("Ollama Model", model_options, index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)

api_key = ""
endpoint = ""

if provider in ["OpenAI", "Anthropic"]:
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    endpoint = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")

max_opt = st.sidebar.number_input("How many attempt to generate RDF/SHACL data?", 1, 10, 3, help="Number of times the LLM should attempt to generate RDF/SHACL")
max_corr = st.sidebar.number_input("How many attempt to correct RDF/SHACL data to pass the validation process?", 1, 10, 3, help="Number of times the LLM should attempt to fix RDF/SHACL after validation fails")

# Read content from a local text file
with open("BAM_Creep.txt", "r") as file:
    file_content = file.read()

# Input section
st.subheader("🔬 Input Test Data")
uploaded_file = st.file_uploader("Upload a file with mechanical test data", type=["txt", "csv", "json", "lis"])
example = st.checkbox("Use example input")
if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")
elif example:
    user_input = st.text_area("Mechanical Test Description:", value = file_content, height=300)
else:
    user_input = st.text_area("Mechanical Test Description:", placeholder="Paste mechanical test data here...", height=200)

if st.button("Generate RDF & SHACL") and user_input.strip():
    with st.spinner("Running Agent-Based Pipeline..."):
        model_info = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "endpoint": endpoint
        }

        agent = SemanticPipelineAgent(model_info, max_opt, max_corr)
        # Manual pipeline steps for visibility
        st.subheader("🛠️ Generation & Optimization Passes")
        rdf_code, shacl_code = agent.generator.run(user_input)
        st.markdown("### 🟢 Initial RDF Output")
        st.code(rdf_code, language="turtle")
        st.markdown("### 🟢 Initial SHACL Output")
        st.code(shacl_code, language="turtle")

        for i in range(1, max_opt):
            st.markdown(f"### 🔄 Optimization Pass {i}")
            explanation = agent.critic.run(rdf_code, shacl_code)
            st.markdown(f"### 🧠 Critique Explanation (Pass {i})")
            st.info(explanation)
            rdf_code, shacl_code = agent.generator.run(
                user_input, f"{rdf_code} {shacl_code} {explanation}"
            )
            st.markdown(f"### 🟡 Optimized RDF v{i}")
            st.code(rdf_code, language="turtle")
            st.markdown(f"### 🟡 Optimized SHACL v{i}")
            st.code(shacl_code, language="turtle")

        valid, report = agent.validator.run(rdf_code, shacl_code)
        
        correction_attempt = 0
        while not valid and correction_attempt < max_corr:
            correction_attempt += 1
            st.warning(f"❌ SHACL Validation Failed. Attempting correction #{correction_attempt}/{max_corr}")
            with st.expander(f"📋 Validation Report (Attempt {correction_attempt})"):
                st.code(report)

            rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, report)
            valid, report = agent.validator.run(rdf_code, shacl_code)

        mappings = agent.ontology_mapper.run(user_input)

        st.subheader("📄 RDF Output")
        st.code(rdf_code, language="turtle")

        st.subheader("🛡️ SHACL Output")
        st.code(shacl_code, language="turtle")

        if valid:
            st.success("✅ SHACL Validation Passed")
        else:
            st.error("❌ SHACL Validation Failed")
        with st.expander("📋 Validation Report"):
            st.code(report)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download RDF", rdf_code, "validated_mechanical_test.ttl", "text/turtle")
        with col2:
            st.download_button("⬇️ Download SHACL", shacl_code, "validated_mechanical_test_shapes.ttl", "text/turtle")

        # RDF Graph Statistics
        # st.subheader("📊 RDF Model Summary")
        # try:
        #     temp_graph = Graph()
        #     temp_graph.parse(data=rdf_code, format="turtle")
        #     st.metric("Triples", len(temp_graph))
        #     st.metric("Subjects", len(set(temp_graph.subjects())))
        #     st.metric("Predicates", len(set(temp_graph.predicates())))
        #     st.metric("Objects", len(set(temp_graph.objects())))
        # except Exception:
        #     st.info("Could not parse RDF for stats.")

        # Ontology Mappings
        st.subheader("🔎 Suggested Ontology Terms")
        st.markdown(mappings)

        # Visualize RDF
        st.subheader("🌐 RDF Graph Visualization")
        def visualize_rdf(rdf_text):
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

            tmp_dir = tempfile.mkdtemp()
            html_path = os.path.join(tmp_dir, f"graph_{uuid.uuid4()}.html")
            net.save_graph(html_path)
            return html_path

        html_file = visualize_rdf(rdf_code)
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, width=1200, scrolling=True)

        # Add instructions for graph interaction
        st.markdown("""
        ### Graph Navigation Instructions:
        - **Zoom**: Use mouse wheel or pinch gesture
        - **Pan**: Click and drag empty space
        - **Move nodes**: Click and drag nodes to rearrange
        - **View details**: Hover over nodes or edges for full information
        - **Select multiple**: Hold Ctrl or Cmd while clicking nodes
        - **Reset view**: Double-click on empty space
        """)
