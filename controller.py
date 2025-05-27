from agents import RDFGeneratorAgent, ValidatorAgent, CritiqueAgent, OntologyMapperAgent

class SemanticPipelineAgent:
    def __init__(self, model_info, max_optimization=3, max_correction=3):
        self.generator = RDFGeneratorAgent(model_info)
        self.validator = ValidatorAgent()
        self.critic = CritiqueAgent(model_info)
        self.ontology_mapper = OntologyMapperAgent(model_info)
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
            correction_prompt = f"Fix the following RDF/SHACL so it passes validation:\nErrors:\n{report}"
            rdf_code, shacl_code = self.generator.run(correction_prompt)
            conforms, report = self.validator.run(rdf_code, shacl_code)

        # Step 4: Ontology term suggestion
        ontology_mappings = self.ontology_mapper.run(user_input)

        return rdf_code, shacl_code, conforms, report, ontology_mappings
