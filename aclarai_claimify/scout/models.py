# models.py
"""
Data models for the Data Scout agent.
"""
import pydantic


class FitnessReport(pydantic.BaseModel):
    """A structured report from the FitnessAgent evaluating a source document."""
    passed: bool = pydantic.Field(description="True if the document is approved, False if it is rejected.")
    reason: str = pydantic.Field(description="A brief justification for the decision, explaining why the document was approved or rejected.")