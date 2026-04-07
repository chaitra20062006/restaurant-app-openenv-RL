"""
models.py
Pydantic typed models for the Restaurant Table Management RL environment.
All API request/response objects are validated here.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TableStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED  = "occupied"
    RESERVED  = "reserved"
    COMBINED  = "combined"   # table merged with adjacent table


class Action(str, Enum):
    ASSIGN_TABLE    = "assign_table"
    REJECT_CUSTOMER = "reject_customer"
    DELAY_SEATING   = "delay_seating"
    COMBINE_TABLES  = "combine_tables"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Table(BaseModel):
    """Represents a single restaurant table."""
    id: int                            = Field(..., description="Unique table identifier")
    capacity: int                      = Field(..., ge=2, description="Max seats at this table")
    status: TableStatus                = TableStatus.AVAILABLE
    time_seated: int                   = Field(0,  ge=0, description="Steps since last customer was seated")
    party_size: int                    = Field(0,  ge=0, description="Current occupying party size (0 if empty)")
    combined_with: Optional[int]       = Field(None, description="ID of table this is combined with")
    revenue_earned: float              = Field(0.0, ge=0.0, description="Revenue from this table in episode")

    class Config:
        use_enum_values = True


class Customer(BaseModel):
    """A customer/party waiting in the queue."""
    id: int                            = Field(..., description="Unique customer identifier")
    party_size: int                    = Field(..., ge=1, le=12, description="Number of people in party")
    patience_remaining: int            = Field(..., ge=0, description="Steps before customer walks out")
    arrival_step: int                  = Field(..., ge=0, description="Time step when customer arrived")
    revenue_value: float               = Field(..., ge=0.0, description="Revenue if successfully seated")

    class Config:
        use_enum_values = True


class ObservationState(BaseModel):
    """Full observable state of the environment at a time step."""
    tables: List[Table]
    waiting_queue: List[Customer]
    time_step: int                     = Field(..., ge=0)
    occupancy_rate: float              = Field(..., ge=0.0, le=1.0)
    total_seated: int                  = Field(..., ge=0)
    total_rejected: int                = Field(..., ge=0)
    total_walkouts: int                = Field(..., ge=0)
    total_revenue: float               = Field(..., ge=0.0)
    episode_done: bool                 = False


# ---------------------------------------------------------------------------
# API Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: TaskDifficulty = TaskDifficulty.EASY
    seed: Optional[int]  = None          # Override yaml seed if provided


class ResetResponse(BaseModel):
    status: str          = "ok"
    task: str
    observation: ObservationState
    message: str         = ""


class StepRequest(BaseModel):
    action: Action
    table_id: Optional[int]    = None   # For assign_table / combine_tables
    combine_with: Optional[int] = None  # Second table for combine_tables


class StepResponse(BaseModel):
    observation: ObservationState
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    observation: ObservationState
    episode_active: bool


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Grader models
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    score: float                       = Field(..., ge=0.0, le=1.0)
    efficiency_score: float            = Field(..., ge=0.0, le=1.0)
    revenue_score: float               = Field(..., ge=0.0, le=1.0)
    satisfaction_score: float          = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any]           = Field(default_factory=dict)
