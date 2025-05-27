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

max_opt = st.sidebar.number_input("Optimization Attempts", 1, 5, 3)
max_corr = st.sidebar.number_input("Correction Attempts", 1, 5, 3)

# Input section
st.subheader("🔬 Input Test Data")
uploaded_file = st.file_uploader("Upload a file with mechanical test data", type=["txt", "csv", "json", "lis"])
example = st.checkbox("Use example input")
if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")
elif example:
    user_input = st.text_area("Mechanical Test Description:", value="""BAM 5.2 Vh5205_C-95.LIS						
------------------------------------						
ENTRY	SYMBOL	UNIT		* Information common to all tests		
Date of test start			30.8.23 9:06 AM			
Test ID			Vh5205_C-95			
Test standard			DIN EN ISO 204:2019-4	*		
Specified temperature	T	?	980 °C	*		
Type of loading			Tension	*		
Initial stress	Ro	MPa	140			
(Digital) Material Identifier			CMSX-6	*		
"Description of the manufacturing process - as-tested material
"			Single Crystal Investment Casting from a Vacuum Induction Refined Ingot and subsequent Heat Treatment (annealed and aged).	*		
Single crystal orientation		°	7,5			
Type of test piece II			Round cross section	*		
Type of test piece III			Smooth test piece	*		
Sensor type - Contacting extensometer			Clip-on extensometer	*		
Min. test piece diameter at room temperature	D	mm	5,99			
Reference length for calculation of percentage elongations	Lr = Lo	mm	23,9			
Reference length for calculation of percentage extensions	Lr = Le	mm	22,9			
Heating time		h	1,61			
Soak time before the test		h	2,81			
Test duration	t	h	1010			
Creep rupture time	tu	h	Not applicable			
Percentage permanent elongation	Aper	%	1,14			
Percentage elongation after creep fracture	Au	%	Not applicable			
Percentage reduction of area after creep fracture	Zu	%	Not applicable			
Percentage total extension	et	%	0,964			
Percentage initial total extension	eti	%	0,153			
Percentage elastic extension	ee	%	0,153			
Percentage initial plastic extension	ei	%	0			
Percentage plastic extension	ep	%	0,811			
Percentage creep extension	ef	%	0,811""")
else:
    user_input = st.text_area("Mechanical Test Description:", placeholder="Paste mechanical test data here...")

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
        st.subheader("📊 RDF Model Summary")
        try:
            temp_graph = Graph()
            temp_graph.parse(data=rdf_code, format="turtle")
            st.metric("Triples", len(temp_graph))
            st.metric("Subjects", len(set(temp_graph.subjects())))
            st.metric("Predicates", len(set(temp_graph.predicates())))
            st.metric("Objects", len(set(temp_graph.objects())))
        except Exception:
            st.info("Could not parse RDF for stats.")

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
            net = Network(height="900px", width="100%", directed=True)
            net.repulsion(node_distance=300, spring_length=300)
            for node in nx_graph.nodes:
                label = node.split("/")[-1].split("#")[-1]
                is_blank = bool(re.match(r'^n\\d+$', label))
                net.add_node(node, label="" if is_blank else label, size=15 if is_blank else 25, color="#E8E8E8" if is_blank else "#97C2FC", font={'size': 16}, title=node)
            for u, v, d in nx_graph.edges(data=True):
                label = d["label"].split("/")[-1].split("#")[-1]
                net.add_edge(u, v, label=label, font={'size': 12}, title=d["label"])
            tmp_dir = tempfile.mkdtemp()
            path = os.path.join(tmp_dir, f"graph_{uuid.uuid4()}.html")
            net.save_graph(path)
            return path

        html_file = visualize_rdf(rdf_code)
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, width=1200, scrolling=True)
