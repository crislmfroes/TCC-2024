from pydantic import BaseModel

class UpdateResponseModel(BaseModel):
    reasoning: str
    observation: str
    available_actions: list[str]
    task_done: bool

class EvaluationResponseModel(BaseModel):
    reasoning: str
    score: int

class ProposalResponseModel(BaseModel):
    thought: str
    action: str