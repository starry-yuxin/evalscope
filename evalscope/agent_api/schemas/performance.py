"""Contracts for the framework-independent performance-test backend."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from evalscope.agent_api.contracts import AgentApiModel


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


class PerformanceTarget(AgentApiModel):
    """Reference to an OpenAI-compatible model endpoint under load."""

    model_id: str
    api_url: str
    credential_ref: str
    api: str = 'openai'
    tokenizer_path: Optional[str] = None


class PerformanceTestConfig(AgentApiModel):
    """Bounded performance-test parameters accepted from the unified API."""

    target: PerformanceTarget
    dataset: str = 'openqa'
    dataset_path: Optional[str] = None
    dataset_args: Optional[Dict[str, Any]] = None
    number: int | List[int] = 100
    parallel: int | List[int] = 1
    rate: float | List[float] = -1
    open_loop: bool = False
    duration: Optional[float] = Field(default=None, gt=0, le=1800)
    warmup_num: int | float = Field(default=0, ge=0)
    stream: bool = True
    max_tokens: int = Field(default=128, ge=1, le=2048)
    min_tokens: Optional[int] = Field(default=None, ge=0, le=2048)
    min_prompt_length: int = Field(default=0, ge=0, le=16384)
    max_prompt_length: int = Field(default=2048, ge=1, le=16384)
    prompt: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    output_dir: str = 'outputs/performance'
    no_timestamp: bool = False
    no_test_connection: bool = False
    seed: Optional[int] = 42

    @field_validator('number', mode='before')
    @classmethod
    def _validate_number(cls, value: Any) -> Any:
        values = _as_list(value)
        if not values or any(
            isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 1000 for item in values
        ):
            raise ValueError('number values must be integers in [1, 1000]')
        return value

    @field_validator('parallel', mode='before')
    @classmethod
    def _validate_parallel(cls, value: Any) -> Any:
        values = _as_list(value)
        if not values or any(
            isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 32 for item in values
        ):
            raise ValueError('parallel values must be integers in [1, 32]')
        return value

    @field_validator('rate', mode='before')
    @classmethod
    def _validate_rate(cls, value: Any) -> Any:
        values = _as_list(value)
        if not values or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise ValueError('rate values must be numbers')
        if any(item != -1 and not 0 < item <= 100 for item in values):
            raise ValueError('rate values must be -1 or in (0, 100]')
        return value

    @model_validator(mode='after')
    def _validate_cross_fields(self) -> 'PerformanceTestConfig':
        numbers = _as_list(self.number)
        sweep = _as_list(self.rate if self.open_loop else self.parallel)
        if len(numbers) != len(sweep):
            mode = 'rate' if self.open_loop else 'parallel'
            raise ValueError(f'number and {mode} must have the same number of sweep values')
        if self.open_loop and any(rate <= 0 for rate in _as_list(self.rate)):
            raise ValueError('open-loop rate values must be in (0, 100]')
        if self.min_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError('min_tokens must be less than or equal to max_tokens')
        if self.min_prompt_length > self.max_prompt_length:
            raise ValueError('min_prompt_length must be less than or equal to max_prompt_length')
        return self


class PerformanceArtifacts(AgentApiModel):
    """EvalScope performance output location."""

    output_dir: str


class PerformanceProvenance(AgentApiModel):
    """Non-secret execution metadata for reproducibility."""

    evalscope_version: str
    model_id: str
    api_url: str
    api: str
    dataset: str


class PerformanceTestResult(AgentApiModel):
    """Structured performance result returned to the unified task lifecycle."""

    task_type: Literal['performance'] = 'performance'
    summary: Dict[str, Any]
    artifacts: PerformanceArtifacts
    provenance: PerformanceProvenance
