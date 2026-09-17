from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class System(str, Enum):
    DATABASE = "Database"
    FRONTEND = "Frontend"
    NETWORK = "Network"
    UNKNOWN = "Unknown"

class Severity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class Action(str, Enum):
    RESTART = "Restart"
    SCALE = "Scale"
    PATCH = "Patch"
    INVESTIGATE = "Investigate"

class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid",frozen=True)

    system : System
    severity: Severity
    action: Action
    confidence: float | None = Field(ge=0,le=1)

class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id : str
    complaint: str = Field(min_length=1,max_length=4000)
    source: str | None = None
    received_at: datetime | None = None

    @field_validator('incident_id')
    @classmethod
    def validate_id(cls, value):
        if not value.strip():
            raise ValueError("incident id must not be blank")
        return value.strip()

    @field_validator('complaint')
    @classmethod
    def validate_complaint(cls, value):
        if not value.strip():
            raise ValueError("complaint must not be blank")
        return value.strip()

    @field_validator("received_at")
    @classmethod
    def validate_date(cls,value):
        if value is not None and value.tzinfo is None:
            raise ValueError("received_at must be timezone-aware")
        return value

class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens : int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

class TriageErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    BAD_REQUEST = "bad_request"
    AUTHENTICATION = "authentication"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    SCHEMA_VALIDATION = "schema_validation"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: TriageErrorType
    message: str = Field(min_length=1)
    retryable: bool
    request_id: str | None = None

class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    complaint: str
    status: Literal["success","error"]
    decision: TriageDecision | None = None
    error: ErrorInfo | None = None
    raw_response: str | None = None
    usage: TokenUsage
    model: str
    attempts: int = Field(ge=1)
    latency_ms : int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_model(self):
        if self.status == "success":
            if self.decision is None:
                raise ValueError("Decision cannot be empty if status is success")
            if self.error is not None:
                raise ValueError("Error cannot be not null if status is success")
        elif self.status == "error":
            if self.decision is not None:
                    raise ValueError("Decision cannot be not null if status is error")
            if self.error is None:
                    raise ValueError("Error cannot be null or empty if status is error")

        return self


class BatchJobStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    provider_batch_id: str | None = None
    status: BatchJobStatus
    total: int = Field(ge=0)
    succeeded: int = Field(default=0,ge=0)
    failed: int = Field(default=0,ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_model(self):
        if self.succeeded + self.failed > self.total:
            raise ValueError("Succeeded jobs + Failed jobs cannot exceed total jobs")
        if self.status == "completed":
            if self.completed_at is None:
                raise ValueError("Completed at cannot be null for a succeeded job")
            if self.succeeded + self.failed != self.total:
                raise ValueError("Succeeded jobs + Failed jobs should equal total jobs")
        return self


__all__ = [
    "Action",
    "BatchJob",
    "BatchJobStatus",
    "ErrorInfo",
    "Severity",
    "System",
    "TokenUsage",
    "TriageDecision",
    "TriageErrorType",
    "TriageRequest",
    "TriageResult",
]