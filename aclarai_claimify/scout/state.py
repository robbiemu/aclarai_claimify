from typing import TypedDict, List, Dict

class DataScoutState(TypedDict):
    """
    Represents the state of the Data Scout agent.
    """
    mission_goal: str
    deconstructed_goal: Dict
    search_plan: Dict
    search_queries: List[str]
    search_results: List[Dict]
    extraction_results: List[str]
    fitness_check_results: List[Dict]
    archived_data: List[Dict]
    iteration: int
    max_iterations: int
