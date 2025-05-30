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
import openai
from anthropic import AuthenticationError as AnthropicAuthError

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
    # model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
    model_options = ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest", "claude-opus-4-20250514"]
    model = st.sidebar.selectbox("Claude Model", model_options, index=0)
else:
    model_options = ["llama3.3:70b-instruct-q8_0", "qwen3:32b-q8_0", "phi4-reasoning:14b-plus-fp16" , "mistral-small3.1:24b-instruct-2503-q8_0"]
    model = st.sidebar.selectbox("Ollama Model", model_options, index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)

api_key = ""
endpoint = ""

if provider in ["OpenAI", "Anthropic"]:
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    endpoint = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")

max_opt = st.sidebar.number_input("How many attempt to optimize RDF/SHACL data?", 1, 10, 3, help="Number of times the LLM should attempt to optimize RDF/SHACL")
max_corr = st.sidebar.number_input("How many attempt to correct RDF/SHACL data to pass the validation process?", 1, 10, 3, help="Number of times the LLM should attempt to fix RDF/SHACL after validation fails")

# Read content from a local text file
try:
    with open("BAM_Creep.txt", "r") as file:
        file_content = file.read()
except FileNotFoundError:
    file_content = "Example file not found"

# Input section
st.subheader("🔬 Input Test Data")
uploaded_file = st.file_uploader("Upload a file with mechanical test data", type=["txt", "csv", "json", "lis"])
example = st.checkbox("Use example input")
# Initialize user_input to avoid NameError
user_input = ""

if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")
elif example:
    user_input = st.text_area("Mechanical Test Description:", value = file_content, height=300)
else:
    user_input = st.text_area("Mechanical Test Description:", placeholder="Paste mechanical test data here...", height=200)


def safe_execute(func, *args, **kwargs):
    """Safely execute a function and handle all exceptions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Log the error for debugging (optional)
        print(f"Error: {e}")  # This will only show in your console, not to users
        raise e  # Re-raise to be caught by your main try-catch

if st.button("Generate RDF & SHACL"):

     # Check for input first
    if not user_input.strip():
        st.error("Please provide some input data to generate RDF and SHACL.")
        st.stop()
    if provider == "OpenAI" and not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif provider == "Anthropic" and not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif provider == "Ollama" and not endpoint:
        st.error("Please enter your Ollama API endpoint in the sidebar.")

    else:
        with st.spinner("Running Agent-Based Pipeline..."):
            model_info = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "api_key": api_key,
                "endpoint": endpoint
            }

            agent = SemanticPipelineAgent(model_info, max_opt, max_corr)
            # Show generation process in an expander to keep it organized
            with st.expander("🛠️ Generation & Optimization Process", expanded=False):
                st.subheader("Initial Generation")
                rdf_code, shacl_code = agent.generator.run(user_input)
                st.markdown("**Initial RDF Output:**")
                st.code(rdf_code, language="turtle")
                st.markdown("**Initial SHACL Output:**")
                st.code(shacl_code, language="turtle")

                # Optimization passes
                for i in range(max_opt):
                    st.markdown(f"### 🔄 Optimization Pass {i+1}")
                    explanation = agent.critic.run(rdf_code, shacl_code)
                    st.markdown(f"**Critique Explanation:**")
                    st.info(explanation)
                    rdf_code, shacl_code = agent.generator.run(
                        user_input, f"{rdf_code} {shacl_code} {explanation}"
                    )
                    st.markdown(f"**Optimized RDF (Pass {i+1}):**")
                    st.code(rdf_code, language="turtle")
                    st.markdown(f"**Optimized SHACL (Pass {i+1}):**")
                    st.code(shacl_code, language="turtle")

            # Validation and correction process
            st.subheader("🔍 Validation & Correction Process")
            valid, report = agent.validator.run(rdf_code, shacl_code)
            
            correction_attempt = 0
            correction_history = []
            
            while not valid and correction_attempt < max_corr:
                correction_attempt += 1
                st.warning(f"❌ SHACL Validation Failed. Attempting correction #{correction_attempt}/{max_corr}")
                
                # Store correction history
                correction_history.append({
                    'attempt': correction_attempt,
                    'rdf': rdf_code,
                    'shacl': shacl_code,
                    'report': report
                })
                
                with st.expander(f"📋 Validation Report (Attempt {correction_attempt})", expanded=False):
                    st.code(report)

                rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, report)
                valid, report = agent.validator.run(rdf_code, shacl_code)

            # Show correction history if there were corrections
            if correction_history:
                with st.expander("🔧 Correction History", expanded=False):
                    for correction in correction_history:
                        st.markdown(f"**Correction Attempt {correction['attempt']}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("*RDF before correction:*")
                            st.code(correction['rdf'], language="turtle")
                        with col2:
                            st.markdown("*SHACL before correction:*")
                            st.code(correction['shacl'], language="turtle")

            # Final validation status
            if valid:
                st.success("✅ Final Validation: PASSED")
            else:
                st.error("❌ Final Validation: FAILED")
                st.warning("⚠️ The generated RDF/SHACL did not pass validation after all correction attempts.")

        # FINAL RESULTS SECTION - Make this very clear
        st.markdown("---")
        st.header("🎯 FINAL VALIDATED RESULTS")

        # Validation status box
        if valid:
            st.success("✅ **STATUS: VALIDATION PASSED** - These are your final, validated RDF and SHACL files.")
        else:
            st.error("❌ **STATUS: VALIDATION FAILED** - These files contain validation errors.")

        # Final RDF output with clear labeling
        st.subheader("📄 Final RDF Output")
        st.markdown("**This is your final RDF file:**")
        st.code(rdf_code, language="turtle")

        # Final SHACL output with clear labeling  
        st.subheader("🛡️ Final SHACL Output")
        st.markdown("**This is your final SHACL shapes file:**")
        st.code(shacl_code, language="turtle")

        # Final validation report
        st.subheader("📋 Final Validation Report")
        with st.expander("View Final Validation Details", expanded=valid is False):
            st.code(report)

        # Download buttons with clear labeling
        st.subheader("⬇️ Download Final Files")
        col1, col2 = st.columns(2)
        with col1:
            download_filename_rdf = f"final_rdf_{'validated' if valid else 'with_errors'}.ttl"
            st.download_button(
                "📥 Download Final RDF", 
                rdf_code, 
                download_filename_rdf, 
                "text/turtle",
                help="Download the final RDF file (validated)" if valid else "Download the final RDF file (contains validation errors)"
            )
        with col2:
            download_filename_shacl = f"final_shacl_{'validated' if valid else 'with_errors'}.ttl"
            st.download_button(
                "📥 Download Final SHACL", 
                shacl_code, 
                download_filename_shacl, 
                "text/turtle",
                help="Download the final SHACL file (validated)" if valid else "Download the final SHACL file (contains validation errors)"
            )

        # Ontology Mappings
        st.subheader("🔎 Suggested Ontology Terms")
        mappings = agent.ontology_mapper.run(user_input)
        st.markdown(mappings)

        # Visualize FINAL RDF
        st.subheader("🌐 Final RDF Graph Visualization")
        st.markdown("**Visualization of your final RDF data:**")

            # Visualize RDF
            # st.subheader("🌐 RDF Graph Visualization")
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

                # tmp_dir = tempfile.mkdtemp()
                # html_path = os.path.join(tmp_dir, f"graph_{uuid.uuid4()}.html")
                # net.save_graph(html_path)
                # return html_path
                # Generate HTML content directly instead of saving to file
                html_content = net.generate_html()
                return html_content
            
            # except Exception as e:
            #     st.error(f"Failed to parse RDF: {e}")
            #     return None
            # Handle different types of errors
            except (openai.error.OpenAIError, AnthropicAuthError) as e:
                if isinstance(e, openai.AuthenticationError):
                    st.error("❌ **Invalid OpenAI API Key**\n\nThe provided OpenAI API key is incorrect. Please verify your API key at: https://platform.openai.com/account/api-keys")
                elif isinstance(e, openai.RateLimitError):
                    st.error("❌ **OpenAI Rate Limit Exceeded**\n\nYou're making requests too quickly. Please wait a moment and try again.")
                elif isinstance(e, openai.InsufficientQuotaError):
                    st.error("❌ **OpenAI Quota Exceeded**\n\nYou have exceeded your API usage quota. Please check your billing settings.")
                else:
                    # Check error message for other common issues
                    error_message = str(e).lower()
                    if "authentication" in error_message or "401" in error_message or "invalid_api_key" in error_message:
                        if provider == "OpenAI":
                            st.error("❌ **Invalid OpenAI API Key**\n\nPlease check your API key and try again.")
                        elif provider == "Anthropic":
                            st.error("❌ **Invalid Anthropic API Key**\n\nPlease check your API key and try again.")
                    elif provider == "Ollama" and ("connection" in error_message or "refused" in error_message):
                        st.error(f"❌ **Cannot Connect to Ollama**\n\nFailed to connect to Ollama at `{endpoint}`. Please make sure Ollama is running and accessible.")
                    else:
                        st.error(f"❌ **An Error Occurred**\n\nSomething went wrong: {type(e).__name__}")
                
                # Optional: Show technical details for debugging
                with st.expander("🔧 Technical Details (for debugging)"):
                    st.code(str(e))
            
            # finally:
            # # Clean up temporary files
            #     if 'tmp_dir' in locals():
            #         import shutil
            #         shutil.rmtree(tmp_dir, ignore_errors=True)

        # html_file = visualize_rdf(rdf_code)
        # with open(html_file, 'r', encoding='utf-8') as f:
        #     html_content = f.read()
        # components.html(html_content, height=1000, width=1200, scrolling=True)

        # Replace the code that calls visualize_rdf with this:
        html_content = visualize_rdf(rdf_code)
        if html_content:
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
        else:
            st.error("Could not generate RDF visualization")
