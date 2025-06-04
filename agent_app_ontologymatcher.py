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
from agents import OntologyMatcherAgent
import hashlib
import re


load_dotenv()


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



st.set_page_config(page_title="AgentSem", layout="wide")
st.title("🧠 AgentSem: Agent-Based Semantic Data Generator")


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

st.sidebar.header("🔍 Ontology Matching Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold", 
    0.0, 1.0, 0.7, 0.1,
    help="Minimum similarity score for ontology term matching (0.0 = very loose, 1.0 = exact match only)"
)

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

            # 🔐 Validate API key
            if provider in ["OpenAI", "Anthropic", "Ollama"]:
                valid, message = validate_api_key(provider, api_key, endpoint)
                if not valid:
                    st.error(f"❌ {message}")
                    st.stop()
                    
            agent = SemanticPipelineAgent(model_info, max_opt, max_corr)

            # agent.ontology_matcher = OntologyMatcherAgent()
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

            # First, try basic syntax cleanup
            rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)

            # Check basic Turtle syntax first
            rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
            shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)

            if not rdf_syntax_valid:
                st.error(f"🚫 RDF Syntax Error: {rdf_syntax_error}")
            if not shacl_syntax_valid:
                st.error(f"🚫 SHACL Syntax Error: {shacl_syntax_error}")

            # Only proceed with SHACL validation if basic syntax is valid
            if rdf_syntax_valid and shacl_syntax_valid:
                valid, report = agent.validator.run(rdf_code, shacl_code)
            else:
                valid = False
                report = f"Syntax errors prevent SHACL validation:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"

            correction_attempt = 0
            correction_history = []
            previous_core_errors = []  # Track core error patterns instead of full reports
            consecutive_failures = 0  # Track consecutive correction failures

            while not valid and correction_attempt < max_corr:
                correction_attempt += 1
                
                # Extract core error for loop detection
                core_error = extract_core_error(report)
                
                # More sophisticated retry logic
                if not should_retry_correction(report, previous_core_errors, max_same_error=2):
                    st.warning(f"🔄 **Similar Error Pattern Detected** after {correction_attempt-1} attempts.")
                    st.info("💡 **Trying alternative correction approach...**")
                    
                    # Try one more time with additional context
                    if consecutive_failures < 2:  # Allow 2 alternative attempts
                        consecutive_failures += 1
                        st.info(f"🛠️ Alternative correction attempt #{consecutive_failures}")
                        
                        # Add more context to help the LLM understand the pattern
                        enhanced_report = f"""
                        REPEATED ERROR PATTERN DETECTED: {core_error}
                        
                        Previous attempts have failed to fix this issue. Please:
                        1. Focus on the ROOT CAUSE of this specific error type
                        2. Consider completely different approaches to modeling this data
                        3. Simplify the constraints if they are too restrictive
                        4. Check for fundamental modeling issues
                        
                        Original error report:
                        {report}
                        """
                        
                        try:
                            rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, enhanced_report)
                            rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)
                            
                            # Validate syntax first
                            rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
                            shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)
                            
                            if rdf_syntax_valid and shacl_syntax_valid:
                                valid, report = agent.validator.run(rdf_code, shacl_code)
                                if valid:
                                    st.success("✨ Alternative correction approach successful!")
                                    break
                            else:
                                valid = False
                                report = f"Syntax errors after correction:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"
                                
                        except Exception as e:
                            st.error(f"Error during alternative correction: {str(e)}")
                            break
                    else:
                        st.error("❌ Multiple correction approaches failed. Breaking correction loop.")
                        st.info("""
                        **💡 Suggestions for manual review:**
                        - The error pattern suggests a fundamental modeling issue
                        - Consider simplifying the SHACL constraints
                        - Review the RDF structure for compliance with expected patterns
                        - Check if the input data matches the intended semantic model
                        """)
                        break
                
                # Track the core error pattern
                previous_core_errors.append(core_error)
                
                # Determine error type for better user feedback
                error_type = 'syntax' if is_syntax_error(report) else 'validation'
                
                if error_type == 'syntax':
                    st.warning(f"🔤 Syntax Error Detected. Attempting correction #{correction_attempt}/{max_corr}")
                else:
                    st.warning(f"📋 SHACL Validation Failed. Attempting correction #{correction_attempt}/{max_corr}")
                
                # Store correction history with better categorization
                correction_history.append({
                    'attempt': correction_attempt,
                    'rdf': rdf_code,
                    'shacl': shacl_code,
                    'report': report,
                    'error_type': error_type,
                    'core_error': core_error
                })
                
                # Show validation report with better formatting
                with st.expander(f"📋 Validation Report (Attempt {correction_attempt})", expanded=False):
                    st.markdown(f"**Error Type:** {error_type.title()}")
                    st.markdown(f"**Core Error Pattern:** `{core_error}`")
                    st.code(report)
                    
                    # Show problematic sections for syntax errors
                    if error_type == 'syntax' and "line" in report:
                        try:
                            line_match = re.search(r'line (\d+)', report)
                            if line_match:
                                line_num = int(line_match.group(1))
                                lines = rdf_code.split('\n')
                                if line_num <= len(lines):
                                    st.markdown(f"**Problematic line {line_num}:**")
                                    # Show context around the problematic line
                                    start_line = max(0, line_num - 3)
                                    end_line = min(len(lines), line_num + 2)
                                    context = '\n'.join([f"{i+1:3d}: {lines[i]}" for i in range(start_line, end_line)])
                                    st.code(context)
                        except:
                            pass
                
                # Attempt correction with progress feedback
                try:
                    with st.spinner(f"Correcting {error_type} error (attempt {correction_attempt})..."):
                        rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, report)
                        
                        # Apply cleanup after correction
                        rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)
                        
                        # Validate syntax first before SHACL validation
                        rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
                        shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)
                        
                        if rdf_syntax_valid and shacl_syntax_valid:
                            valid, report = agent.validator.run(rdf_code, shacl_code)
                            if valid:
                                consecutive_failures = 0  # Reset failure counter on success
                        else:
                            valid = False
                            report = f"Syntax errors after correction:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"
                            
                except Exception as e:
                    st.error(f"Error during correction attempt {correction_attempt}: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        st.error("Multiple correction failures. Stopping correction process.")
                        break

            # Enhanced correction history display
            if correction_history:
                with st.expander("🔧 Detailed Correction History", expanded=False):
                    # Group by error type for better organization
                    syntax_corrections = [c for c in correction_history if c['error_type'] == 'syntax']
                    validation_corrections = [c for c in correction_history if c['error_type'] == 'validation']
                    
                    if syntax_corrections:
                        st.markdown("### 🔤 Syntax Error Corrections")
                        for correction in syntax_corrections:
                            st.markdown(f"**Attempt {correction['attempt']}** - Core Error: `{correction['core_error']}`")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("*RDF before correction:*")
                                st.code(correction['rdf'][:500] + "..." if len(correction['rdf']) > 500 else correction['rdf'], language="turtle")
                            with col2:
                                st.markdown("*SHACL before correction:*")
                                st.code(correction['shacl'][:500] + "..." if len(correction['shacl']) > 500 else correction['shacl'], language="turtle")
                    
                    if validation_corrections:
                        st.markdown("### 📋 Validation Error Corrections")
                        for correction in validation_corrections:
                            st.markdown(f"**Attempt {correction['attempt']}** - Core Error: `{correction['core_error']}`")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("*RDF before correction:*")
                                st.code(correction['rdf'][:500] + "..." if len(correction['rdf']) > 500 else correction['rdf'], language="turtle")
                            with col2:
                                st.markdown("*SHACL before correction:*")
                                st.code(correction['shacl'][:500] + "..." if len(correction['shacl']) > 500 else correction['shacl'], language="turtle")

            # Enhanced final validation status
            if valid:
                st.success("✅ Final Validation: PASSED")
                if correction_attempt > 0:
                    st.info(f"🎉 Successfully corrected after {correction_attempt} attempt(s)!")
                    
                    # Show what types of errors were fixed
                    error_types_fixed = list(set([c['error_type'] for c in correction_history]))
                    if len(error_types_fixed) == 1:
                        st.info(f"✨ Fixed {error_types_fixed[0]} errors")
                    else:
                        st.info(f"✨ Fixed multiple error types: {', '.join(error_types_fixed)}")
            else:
                st.error("❌ Final Validation: FAILED")
                if correction_attempt >= max_corr:
                    st.warning(f"⚠️ The generated RDF/SHACL did not pass validation after {max_corr} correction attempts.")
                    
                    # Provide specific guidance based on the final error pattern
                    final_core_error = extract_core_error(report)
                    if is_syntax_error(report):
                        st.info("""
                        **💡 Persistent Syntax Error Detected:**
                        
                        The issue appears to be malformed Turtle syntax. Common solutions:
                        - Check for unescaped quotes or special characters
                        - Ensure proper URI formatting (use <> for full URIs)
                        - Verify all statements end with proper punctuation (. ; ,)
                        - Check for missing prefixes or namespace declarations
                        """)
                    elif "constraint violation" in final_core_error:
                        st.info(f"""
                        **💡 Persistent SHACL Constraint Violation:**
                        
                        Core error pattern: `{final_core_error}`
                        
                        Possible solutions:
                        - Review if the SHACL constraints are too restrictive for your data
                        - Check if the RDF structure matches the expected semantic model  
                        - Consider simplifying complex constraints
                        - Verify that required properties are present in the RDF data
                        """)
                    else:
                        st.info(f"""
                        **💡 Persistent Validation Error:**
                        
                        Core error pattern: `{final_core_error}`
                        
                        This suggests a fundamental mismatch between the RDF data structure and SHACL constraints.
                        Consider manual review of both files.
                        """)
                else:
                    st.warning("⚠️ Validation failed due to syntax or other errors.")

            # Add debugging information for developers
            if st.sidebar.checkbox("🔧 Show Debug Info", help="Show technical details for debugging"):
                with st.expander("🔧 Debug Information"):
                    st.markdown("**Error Pattern Analysis:**")
                    if previous_core_errors:
                        from collections import Counter
                        error_counts = Counter(previous_core_errors)
                        for error, count in error_counts.items():
                            st.markdown(f"- `{error}`: occurred {count} time(s)")
                    
                    st.markdown("**Correction Attempt Summary:**")
                    st.markdown(f"- Total attempts: {correction_attempt}")
                    st.markdown(f"- Consecutive failures: {consecutive_failures}")
                    st.markdown(f"- Final validation status: {'PASSED' if valid else 'FAILED'}")
                
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

        # NEW: Ontology Matching Analysis
        st.subheader("🎯 Ontology Term Matching Analysis")
        st.markdown("**Analysis of RDF terms against local ontology files:**")

        with st.spinner("Analyzing ontology matches..."):
            try:
                ontology_analysis = agent.ontology_matcher.run(rdf_code, similarity_threshold)
                st.markdown(ontology_analysis)
            except Exception as e:
                st.error(f"Error during ontology matching: {str(e)}")
                st.info("💡 **Tip:** Make sure to place your ontology files (.ttl or .owl) in an 'ontologies/' directory in the same location as this application.")

        # Add instructions for users
        st.sidebar.info("""
        📁 **Ontology Files Setup:**

        1. Create an 'ontologies/' directory
        2. Place your .ttl or .owl files there
        3. The app will automatically load them for matching analysis
        """)

        # Add file uploader for ontology files in sidebar
        st.sidebar.header("📁 Upload Ontology Files")
        uploaded_ontology = st.sidebar.file_uploader(
            "Upload ontology files", 
            type=["ttl", "owl"], 
            accept_multiple_files=True,
            help="Upload .ttl or .owl ontology files for matching analysis"
        )
        if uploaded_ontology:
            # Create ontologies directory if it doesn't exist
            os.makedirs("ontologies", exist_ok=True)
            
            for file in uploaded_ontology:
                file_path = os.path.join("ontologies", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                st.sidebar.success(f"✅ Uploaded: {file.name}")
            
            # Reload the ontology matcher to include new files
            agent.ontology_matcher = OntologyMatcherAgent()
            st.sidebar.info("🔄 Ontology files loaded. Run the pipeline to see matching results.")

        original_rdf = rdf_code
        original_shacl = shacl_code

        # NEW: Apply Ontology Term Replacements
        st.subheader("🔄 Ontology Term Replacement")
        st.markdown("**Applying exact matches from ontology analysis:**")

        with st.spinner("Applying ontology term replacements..."):
            try:
                # Apply replacements
                replaced_rdf, replaced_shacl, replacement_report, replacement_validation = agent.apply_ontology_replacements(
                    rdf_code, shacl_code, similarity_threshold
                )
                
                # Show replacement report
                st.markdown(replacement_report)
                
                # If replacements were made, update the final codes and validate
                if replacement_validation and replacement_validation["replacements_made"] > 0:
                    st.success(f"✅ Applied {replacement_validation['replacements_made']} exact term replacements!")
                    
                    # Update the final codes
                    rdf_code = replaced_rdf
                    shacl_code = replaced_shacl
                    
                    # Show validation results for replaced version
                    if replacement_validation["conforms"]:
                        st.success("✅ Replaced RDF/SHACL passes validation!")
                        valid = True  # Update validation status
                    else:
                        st.warning("⚠️ Replaced RDF/SHACL has validation issues:")
                        with st.expander("View Replacement Validation Report"):
                            st.code(replacement_validation["report"])
                        
                        # Offer to use original or replaced version
                        use_replaced = st.radio(
                            "Which version would you like to use as final?",
                            ["Use replaced version (with ontology terms)", "Use original version (before replacement)"],
                            index=0
                        )
                        
                        if use_replaced == "Use original version (before replacement)":
                            # Restore original codes
                            rdf_code = replaced_rdf  # This should be swapped back to original
                            shacl_code = replaced_shacl  # This should be swapped back to original
                            st.info("Using original version as final output.")
                else:
                    st.info("ℹ️ No exact matches found for replacement. Using original RDF/SHACL.")
                    
            except Exception as e:
                st.error(f"Error during ontology replacement: {str(e)}")
                st.info("Using original RDF/SHACL without replacements.")

        if replacement_validation and replacement_validation["replacements_made"] > 0:
            st.subheader("📊 Before/After Comparison")
            
            # Create tabs for comparison
            tab1, tab2 = st.tabs(["🔍 RDF Comparison", "🛡️ SHACL Comparison"])
            
            with tab1:
                st.markdown("**RDF Changes:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("*Before Replacement:*")
                    st.code(rdf_code if 'original_rdf' not in locals() else original_rdf, language="turtle")
                
                with col2:
                    st.markdown("*After Replacement:*")
                    st.code(replaced_rdf, language="turtle")
            
            with tab2:
                st.markdown("**SHACL Changes:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("*Before Replacement:*")
                    st.code(shacl_code if 'original_shacl' not in locals() else original_shacl, language="turtle")
                
                with col2:
                    st.markdown("*After Replacement:*")
                    st.code(replaced_shacl, language="turtle")
        # Replace the existing FINAL RESULTS SECTION with this updated version

        # FINAL RESULTS SECTION - Make this very clear
        st.markdown("---")
        st.header("🎯 FINAL VALIDATED RESULTS")

        # Show replacement status
        replacement_status = ""
        if 'replacement_validation' in locals() and replacement_validation and replacement_validation["replacements_made"] > 0:
            replacement_status = f" (with {replacement_validation['replacements_made']} ontology term replacements)"

        # Validation status box
        if valid:
            st.success(f"✅ **STATUS: VALIDATION PASSED{replacement_status}** - These are your final, validated RDF and SHACL files.")
        else:
            st.error(f"❌ **STATUS: VALIDATION FAILED{replacement_status}** - These files contain validation errors.")

        # Show replacement summary if applicable
        if 'replacement_validation' in locals() and replacement_validation and replacement_validation["replacements_made"] > 0:
            st.info(f"🔄 **Ontology Integration:** {replacement_validation['replacements_made']} terms were replaced with ontology equivalents")

        # Final RDF output with clear labeling
        st.subheader("📄 Final RDF Output")
        st.markdown("**This is your final RDF file" + (" (with ontology term replacements)" if replacement_status else "") + ":**")
        st.code(rdf_code, language="turtle")

        # Final SHACL output with clear labeling  
        st.subheader("🛡️ Final SHACL Output")
        st.markdown("**This is your final SHACL shapes file" + (" (with ontology term replacements)" if replacement_status else "") + ":**")
        st.code(shacl_code, language="turtle")

        # Final validation report
        st.subheader("📋 Final Validation Report")
        with st.expander("View Final Validation Details", expanded=valid is False):
            if 'replacement_validation' in locals() and replacement_validation and replacement_validation["replacements_made"] > 0:
                st.markdown("**Validation of final version (after ontology term replacement):**")
                st.code(replacement_validation["report"] if not replacement_validation["conforms"] else "All SHACL constraints satisfied.")
            else:
                st.code(report)

        # Download buttons with clear labeling
        st.subheader("⬇️ Download Final Files")
        col1, col2 = st.columns(2)
        with col1:
            suffix = "with_ontology_terms" if replacement_status else ("validated" if valid else "with_errors")
            download_filename_rdf = f"final_rdf_{suffix}.ttl"
            st.download_button(
                "📥 Download Final RDF", 
                rdf_code, 
                download_filename_rdf, 
                "text/turtle",
                help=f"Download the final RDF file {replacement_status}"
            )
        with col2:
            download_filename_shacl = f"final_shacl_{suffix}.ttl"
            st.download_button(
                "📥 Download Final SHACL", 
                shacl_code, 
                download_filename_shacl, 
                "text/turtle",
                help=f"Download the final SHACL file {replacement_status}"
            )
        # Visualize FINAL RDF
        st.subheader("🌐 Final RDF Graph Visualization")
        st.markdown("**Visualization of your final RDF data:**")


        def visualize_rdf(rdf_text, max_nodes=100):
            """
            Enhanced RDF visualization with better styling, filtering, and interaction
            """
            try:
                g = Graph().parse(data=rdf_text, format="turtle")
                nx_graph = nx.DiGraph()
                
                # Count total triples for filtering if needed
                total_triples = len(list(g))
                
                # Extract namespaces for better labeling
                namespaces = {}
                for prefix, uri in g.namespaces():
                    if prefix:  # Skip empty prefixes
                        namespaces[str(uri)] = prefix
                
                def get_short_label(uri_str):
                    """Extract meaningful short labels from URIs"""
                    # Check if it's a known namespace
                    for namespace, prefix in namespaces.items():
                        if uri_str.startswith(namespace):
                            return f"{prefix}:{uri_str[len(namespace):]}"
                    
                    # Standard URL/URI processing
                    if "/" in uri_str:
                        label = uri_str.split("/")[-1]
                    elif "#" in uri_str:
                        label = uri_str.split("#")[-1]
                    else:
                        label = uri_str
                        
                    # Clean up common patterns
                    label = label.replace("_", " ").replace("%20", " ")
                    return label[:30] + "..." if len(label) > 30 else label
                
                def get_node_type(uri_str):
                    """Determine node type for styling"""
                    if uri_str.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
                        return "rdf_system"
                    elif uri_str.startswith("http://www.w3.org/2000/01/rdf-schema#"):
                        return "rdfs_system"
                    elif uri_str.startswith("http://www.w3.org/ns/shacl#"):
                        return "shacl_system"
                    elif re.match(r'^n\d+$', uri_str):
                        return "blank_node"
                    elif "http" in uri_str:
                        return "uri_resource"
                    elif uri_str.replace(".", "").replace("-", "").isdigit():
                        return "literal_number"
                    elif len(uri_str) > 50:
                        return "literal_text"
                    else:
                        return "literal_simple"
                
                # Build graph with enhanced node information
                node_info = {}
                edge_counts = {}
                
                for s, p, o in g:
                    s_str, p_str, o_str = str(s), str(p), str(o)
                    
                    # Track edge frequency for styling
                    edge_key = (s_str, o_str, p_str)
                    edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                    
                    # Add nodes with metadata
                    for node_str in [s_str, o_str]:
                        if node_str not in node_info:
                            node_info[node_str] = {
                                'label': get_short_label(node_str),
                                'type': get_node_type(node_str),
                                'full_uri': node_str,
                                'connections': 0
                            }
                        node_info[node_str]['connections'] += 1
                    
                    nx_graph.add_edge(s_str, o_str, predicate=p_str, frequency=edge_counts[edge_key])
                
                # Filter nodes if graph is too large
                if len(nx_graph.nodes) > max_nodes:
                    # Keep most connected nodes
                    sorted_nodes = sorted(node_info.items(), key=lambda x: x[1]['connections'], reverse=True)
                    keep_nodes = set([node for node, _ in sorted_nodes[:max_nodes]])
                    
                    # Create filtered subgraph
                    nx_graph = nx_graph.subgraph(keep_nodes).copy()
                    st.info(f"📊 Graph filtered to show {max_nodes} most connected nodes out of {len(node_info)} total nodes")
                
                # Create enhanced visualization
                net = Network(
                    height="1000px", 
                    width="100%", 
                    directed=True, 
                    notebook=False,
                    bgcolor="#fafafa",
                    font_color="#2c3e50"
                )
                
                # Define node styling based on type
                node_styles = {
                    "rdf_system": {"color": "#e74c3c", "size": 20, "shape": "box"},
                    "rdfs_system": {"color": "#9b59b6", "size": 20, "shape": "box"}, 
                    "shacl_system": {"color": "#f39c12", "size": 20, "shape": "diamond"},
                    "blank_node": {"color": "#bdc3c7", "size": 15, "shape": "dot"},
                    "uri_resource": {"color": "#3498db", "size": 25, "shape": "ellipse"},
                    "literal_number": {"color": "#2ecc71", "size": 18, "shape": "box"},
                    "literal_text": {"color": "#1abc9c", "size": 20, "shape": "text"},
                    "literal_simple": {"color": "#27ae60", "size": 18, "shape": "ellipse"}
                }
                
                # Add nodes with enhanced styling
                for node in nx_graph.nodes():
                    info = node_info.get(node, {})
                    node_type = info.get('type', 'literal_simple')
                    style = node_styles.get(node_type, node_styles['literal_simple'])
                    
                    # Scale size based on connections
                    connection_scale = min(2.0, 1 + info.get('connections', 0) / 10)
                    scaled_size = int(style['size'] * connection_scale)
                    
                    # Special handling for blank nodes
                    if node_type == "blank_node":
                        display_label = ""
                        title_text = f"Blank Node: {node}"
                    else:
                        display_label = info.get('label', node)
                        title_text = f"Type: {node_type.replace('_', ' ').title()}\nConnections: {info.get('connections', 0)}\nFull URI: {info.get('full_uri', node)}"
                    
                    net.add_node(
                        node, 
                        label=display_label,
                        size=scaled_size,
                        color=style['color'],
                        shape=style['shape'],
                        font={'size': 14, 'color': '#2c3e50'},
                        title=title_text,
                        borderWidth=2,
                        borderWidthSelected=4
                    )
                
                # Add edges with enhanced styling
                predicate_colors = {}
                color_palette = ["#34495e", "#7f8c8d", "#95a5a6", "#2c3e50", "#5d6d7e"]
                
                for u, v, data in nx_graph.edges(data=True):
                    predicate = data.get('predicate', '')
                    frequency = data.get('frequency', 1)
                    
                    # Assign colors to predicates
                    if predicate not in predicate_colors:
                        predicate_colors[predicate] = color_palette[len(predicate_colors) % len(color_palette)]
                    
                    edge_label = get_short_label(predicate)
                    edge_width = min(5, 1 + frequency)  # Width based on frequency
                    
                    # Create title with full information
                    edge_title = f"Predicate: {predicate}\nFrequency: {frequency}"
                    
                    net.add_edge(
                        u, v,
                        label=edge_label,
                        color=predicate_colors[predicate],
                        width=edge_width,
                        font={'size': 12, 'color': '#2c3e50', 'strokeWidth': 3, 'strokeColor': '#ffffff'},
                        title=edge_title,
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.2}},
                        smooth={'enabled': True, 'type': 'curvedCW', 'roundness': 0.1}
                    )
                
                # Enhanced physics and layout options
                net.set_options("""
                const options = {
                    "physics": {
                        "enabled": true,
                        "stabilization": {
                            "iterations": 200,
                            "updateInterval": 25,
                            "fit": true
                        },
                        "barnesHut": {
                            "gravitationalConstant": -3000,
                            "springLength": 200,
                            "springConstant": 0.04,
                            "damping": 0.09,
                            "avoidOverlap": 0.1
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
                        "hoverConnectedEdges": true,
                        "selectConnectedEdges": true,
                        "multiselect": true,
                        "tooltipDelay": 100,
                        "zoomView": true,
                        "dragView": true
                    },
                    "manipulation": {
                        "enabled": false
                    },
                    "nodes": {
                        "font": {
                            "size": 14,
                            "face": "Arial, sans-serif"
                        },
                        "borderWidth": 2,
                        "shadow": {
                            "enabled": true,
                            "color": "rgba(0,0,0,0.2)",
                            "size": 5,
                            "x": 2,
                            "y": 2
                        }
                    },
                    "edges": {
                        "font": {
                            "size": 12,
                            "face": "Arial, sans-serif",
                            "strokeWidth": 2,
                            "strokeColor": "#ffffff"
                        },
                        "shadow": {
                            "enabled": true,
                            "color": "rgba(0,0,0,0.1)",
                            "size": 3,
                            "x": 1,
                            "y": 1
                        }
                    }
                }
                """)
                
                # Generate HTML with custom styling
                html_content = net.generate_html()
                
                # Add custom CSS for better appearance
                custom_css = """
                <style>
                #mynetworkid {
                    border: 2px solid #ddd;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    background: linear-gradient(45deg, #f8f9fa 0%, #ffffff 100%);
                }
                .vis-network {
                    outline: none;
                }
                </style>
                """
                
                # Insert custom CSS into HTML
                html_content = html_content.replace('</head>', custom_css + '</head>')
                
                return html_content, len(nx_graph.nodes), len(nx_graph.edges), namespaces
                
            except Exception as e:
                st.error(f"Failed to parse RDF: {e}")
                return None, 0, 0, {}

        # Replace the RDF visualization section with this enhanced version:

        # Visualize FINAL RDF with enhanced features
        st.subheader("🌐 Enhanced RDF Graph Visualization")
        st.markdown("**Interactive visualization of your final RDF data:**")

        # Add visualization controls
        col1, col2, col3 = st.columns(3)
        with col1:
            max_nodes = st.slider("Maximum nodes to display", 50, 500, 150, 25, 
                                help="Limit nodes for better performance with large graphs")
        with col2:
            show_legend = st.checkbox("Show legend", True, help="Display node type legend")
        with col3:
            graph_layout = st.selectbox("Layout style", 
                                    ["Force-directed", "Hierarchical"], 
                                    help="Choose graph layout algorithm")

        # Generate visualization
        html_content, node_count, edge_count, namespaces = visualize_rdf(rdf_code, max_nodes)

        if html_content:
            # Show graph statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", node_count)
            with col2:
                st.metric("Edges", edge_count)
            with col3:
                st.metric("Namespaces", len(namespaces))
            with col4:
                st.metric("Triples", node_count + edge_count)
            
            # Display the visualization
            components.html(html_content, height=1000, width=1200, scrolling=True)
            
            # Show legend if requested
            if show_legend:
                with st.expander("🎨 Visualization Legend", expanded=False):
                    st.markdown("""
                    ### Node Types & Colors:
                    - 🔴 **RDF System** (red boxes): Core RDF vocabulary terms
                    - 🟣 **RDFS System** (purple boxes): RDF Schema vocabulary  
                    - 🟠 **SHACL System** (orange diamonds): SHACL vocabulary terms
                    - 🔵 **URI Resources** (blue ellipses): Your domain resources
                    - 🟢 **Literal Numbers** (green boxes): Numeric values
                    - 🟢 **Literal Text** (teal): Text values
                    - ⚪ **Blank Nodes** (gray dots): Anonymous resources
                    
                    ### Edge Features:
                    - **Width**: Indicates frequency of the relationship
                    - **Color**: Different colors for different predicates
                    - **Direction**: Arrows show triple direction (subject → object)
                    
                    ### Size Scaling:
                    - Larger nodes have more connections
                    - More connected = more important in the graph
                    """)
            
            # Show namespace information
            if namespaces:
                with st.expander("📚 Namespace Prefixes", expanded=False):
                    st.markdown("**Detected namespace prefixes in your RDF:**")
                    for uri, prefix in namespaces.items():
                        st.markdown(f"- **{prefix}:** `{uri}`")
            
            # Enhanced navigation instructions
            with st.expander("🎮 Graph Navigation Guide", expanded=False):
                st.markdown("""
                ### Mouse Controls:
                - **🖱️ Zoom**: Mouse wheel or trackpad pinch
                - **🖱️ Pan**: Click and drag empty space
                - **🖱️ Move nodes**: Click and drag any node
                - **🖱️ Select**: Click nodes or edges to highlight
                - **🖱️ Multi-select**: Hold Ctrl/Cmd while clicking
                
                ### Keyboard Shortcuts:
                - **R**: Reset view to fit all nodes
                - **S**: Take screenshot
                - **Arrows**: Navigate when nodes are selected
                
                ### Visual Features:
                - **Hover effects**: Hover over nodes/edges for details
                - **Connected highlighting**: Click a node to highlight its connections
                - **Smooth transitions**: Animated movements and zoom
                - **Shadow effects**: Enhanced visual depth
                
                ### Performance Tips:
                - Reduce max nodes for better performance with large graphs
                - Use the filter controls to focus on specific parts
                - Reset view (double-click) if graph becomes cluttered
                """)
                
            # Add export options
            st.markdown("### 📤 Export Options")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export Graph Data"):
                    graph_data = {
                        "nodes": node_count,
                        "edges": edge_count,
                        "namespaces": namespaces,
                        "visualization_settings": {
                            "max_nodes": max_nodes,
                            "layout": graph_layout
                        }
                    }
                    st.download_button(
                        "💾 Download Graph Info", 
                        str(graph_data), 
                        "rdf_graph_info.json", 
                        "application/json"
                    )
            
            with col2:
                st.info("💡 **Tip**: Right-click the graph and select 'Save image as...' to export as PNG")

        else:
            st.error("Could not generate RDF visualization")
            st.info("""
            **Possible issues:**
            - Invalid RDF syntax
            - Graph too complex to render
            - Memory limitations
            
            Try reducing the maximum nodes or fixing RDF syntax errors.
            """)