"""
Utility functions for working with prompts in the Claimify pipeline.
Includes functions for generating JSON schemas and formatting prompts.
"""

import json
from typing import Dict, Any, Optional

from .llm_schemas import SelectionResponse, DisambiguationResponse, DecompositionResponse


def get_json_schema_for_stage(stage: str) -> Optional[Dict[str, Any]]:
    """
    Get the JSON schema for a specific pipeline stage.
    
    Args:
        stage: Stage name ("selection", "disambiguation", "decomposition")
        
    Returns:
        JSON schema dictionary or None if stage not found
    """
    schema_models = {
        "selection": SelectionResponse,
        "disambiguation": DisambiguationResponse,
        "decomposition": DecompositionResponse
    }
    
    if stage not in schema_models:
        return None
        
    model = schema_models[stage]
    return model.model_json_schema()


def format_prompt_with_schema(stage: str, prompt_template: str, **kwargs) -> str:
    """
    Format a prompt template with the JSON schema for the stage.
    
    Args:
        stage: Stage name
        prompt_template: Template string with placeholders
        **kwargs: Additional template variables
        
    Returns:
        Formatted prompt string with JSON schema injected
    """
    schema = get_json_schema_for_stage(stage)
    if schema is None:
        # Fallback to original template if schema not available
        return prompt_template.format(**kwargs)
    
    # Add schema to kwargs for template formatting
    kwargs["json_schema"] = json.dumps(schema, indent=2)
    
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        # If template expects json_schema but it's not used, fallback to original
        if "json_schema" in str(e):
            return prompt_template.format(**kwargs)
        raise