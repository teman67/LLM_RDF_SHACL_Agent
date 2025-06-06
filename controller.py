from agents import RDFGeneratorAgent, ValidatorAgent, CritiqueAgent, OntologyMapperAgent, CorrectionAgent, OntologyMatcherAgent

class SemanticPipelineAgent:
    def __init__(self, model_info, max_optimization=3, max_correction=3):
        self.generator = RDFGeneratorAgent(model_info)
        self.validator = ValidatorAgent()
        self.critic = CritiqueAgent(model_info)
        self.ontology_mapper = OntologyMapperAgent(model_info)
        self.corrector = CorrectionAgent(model_info)
        self.ontology_matcher = OntologyMatcherAgent()
        self.max_optimization = max_optimization
        self.max_correction = max_correction

    def run_pipeline(self, user_input):
        # Step 1: Initial generation
        rdf_code, shacl_code = self.generator.run(user_input)

        # Step 2: Optimization loop
        for _ in range(self.max_optimization - 1):
            explanation = self.critic.run(rdf_code, shacl_code)
            rdf_code, shacl_code = self.generator.run(
                user_input, f"{rdf_code}\n{shacl_code}\n\n{explanation}"
            )

        # Step 3: Validation loop
        conforms, report = self.validator.run(rdf_code, shacl_code)
        for _ in range(self.max_correction):
            if conforms:
                break
            correction_prompt = f"""Fix the SHACL validation errors in the following RDF and SHACL data.

                VALIDATION ERRORS:
                {report}

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

                Return the corrected versions now:"""
            rdf_code, shacl_code = self.generator.run(correction_prompt)
            conforms, report = self.validator.run(rdf_code, shacl_code)

        # Step 4: Ontology term suggestion
        ontology_mappings = self.ontology_mapper.run(user_input)

        # Step 5: Ontology matching analysis (NEW)
        ontology_matches = self.ontology_matcher.run(rdf_code)

        return rdf_code, shacl_code, conforms, report, ontology_mappings, ontology_matches
    
    def apply_ontology_replacements(self, rdf_code: str, shacl_code: str, similarity_threshold: float = 0.7):
        """
        Apply ontology term replacements and return updated RDF/SHACL with replacement info
        
        Args:
            rdf_code: Original RDF code
            shacl_code: Original SHACL code
            similarity_threshold: Minimum similarity for matches
            
        Returns:
            Tuple of (updated_rdf, updated_shacl, replacement_report, validation_results)
        """
        # Apply replacements
        updated_rdf, updated_shacl, replacement_list = self.ontology_matcher.replace_exact_matches(
            rdf_code, shacl_code, similarity_threshold
        )
        
        # Generate replacement report
        replacement_report = self.ontology_matcher.generate_replacement_report(replacement_list)
        
        # Validate the updated RDF/SHACL
        validation_results = None
        if replacement_list:  # Only validate if replacements were made
            conforms, report = self.validator.run(updated_rdf, updated_shacl)
            validation_results = {
                "conforms": conforms,
                "report": report,
                "replacements_made": len(replacement_list)
            }
        
        return updated_rdf, updated_shacl, replacement_report, validation_results
