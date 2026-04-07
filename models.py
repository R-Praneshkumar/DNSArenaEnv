"""Typed Pydantic models for the DNS-Env OpenEnv environment."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class Action(BaseModel):
    """Agent action for the DNS debugging environment."""
    command: str = Field(
        ...,
        description=(
            "Command to execute. One of: view_zone, add_record, edit_record, "
            "delete_record, check_zone, dig, submit"
        ),
    )
    args: Dict[str, Any] = Field(default_factory=dict, description="Command arguments")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Environment observation returned to the agent after each step."""
    output: str = Field(default="", description="Command output or feedback text")
    task_description: str = Field(default="", description="Current task description")
    zone_names: List[str] = Field(default_factory=list, description="Available zone file names")
    available_commands: List[str] = Field(
        default_factory=lambda: [
            "view_zone", "add_record", "edit_record",
            "delete_record", "check_zone", "dig", "submit",
        ],
    )
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Episode state tracking."""
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
    task_id: str = ""
    max_steps: int = 30

    class Config:
        extra = "allow"
