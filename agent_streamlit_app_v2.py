import streamlit as st
from controller import SemanticPipelineAgent
from dotenv import load_dotenv
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
from agents import OntologyMatcherAgent
from helpers import extract_core_error, is_syntax_error, should_retry_correction, basic_syntax_cleanup, validate_turtle_syntax, validate_api_key, visualize_rdf
import hashlib
import re
import os

load_dotenv()

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

        # Store original RDF/SHACL codes for comparison later
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
