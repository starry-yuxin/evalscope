"""Contracts for the framework-independent Agent evaluation backend."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from evalscope.agent_api.contracts import AgentApiModel, GenerationOptions, ModelTarget


def _validate_limit(value: Optional[int | float]) -> Optional[int | float]:
    if value is None:
        return value
    if isinstance(value, bool) or value <= 0:
        raise ValueError('limit_per_subset must be positive')
    if isinstance(value, float) and value > 1:
        raise ValueError('a float limit_per_subset must be a fraction in (0, 1]')
    return value


class AgentBenchmarkOptions(AgentApiModel):
    """Per-benchmark dataset selection for an Agent evaluation."""

    subsets: Optional[List[str]] = None
    limit_per_subset: Optional[int | float] = None
    dataset_args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('subsets')
    @classmethod
    def _validate_subsets(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError('subsets must contain at least one non-empty name')
        if len(cleaned) != len(set(cleaned)):
            raise ValueError('subsets must not contain duplicates')
        return cleaned

    @field_validator('limit_per_subset', mode='before')
    @classmethod
    def _validate_benchmark_limit(cls, value: Optional[int | float]) -> Optional[int | float]:
        return _validate_limit(value)

    @model_validator(mode='after')
    def _reject_conflicting_subset_sources(self) -> 'AgentBenchmarkOptions':
        configured = self.dataset_args.get('subset_list')
        if self.subsets is not None and configured is not None and list(configured) != self.subsets:
            raise ValueError('use subsets or dataset_args.subset_list, not both')
        return self


class AgentEvaluationConfig(AgentApiModel):
    """Business parameters consumed by the Agent evaluation backend."""

    target: ModelTarget
    benchmarks: List[str] = Field(min_length=1)
    benchmark_options: Dict[str, AgentBenchmarkOptions] = Field(default_factory=dict)
    limit_per_subset: Optional[int | float] = None
    dataset_hub: Literal['modelscope', 'huggingface'] = 'modelscope'
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    eval_batch_size: int = Field(default=1, ge=1)
    repeats: int = Field(default=1, ge=1)
    ignore_errors: bool = True
    seed: int = 42
    work_dir: str = 'outputs/agent_evaluation'
    use_cache: Optional[str] = None
    rerun_review: bool = False

    @field_validator('benchmarks')
    @classmethod
    def _validate_benchmarks(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError('at least one benchmark is required')
        if len(cleaned) != len(set(cleaned)):
            raise ValueError('benchmarks must not contain duplicates')
        return cleaned

    @field_validator('limit_per_subset', mode='before')
    @classmethod
    def _validate_default_limit(cls, value: Optional[int | float]) -> Optional[int | float]:
        return _validate_limit(value)

    @model_validator(mode='after')
    def _validate_selected_options(self) -> 'AgentEvaluationConfig':
        unknown = sorted(set(self.benchmark_options) - set(self.benchmarks))
        if unknown:
            raise ValueError('benchmark_options contains unselected benchmarks: ' + ', '.join(unknown))
        return self


class AgentEvaluationArtifacts(AgentApiModel):
    """EvalScope output locations produced by the backend."""

    root: str
    benchmark_roots: Dict[str, str]


class AgentEvaluationProvenance(AgentApiModel):
    """Non-secret execution metadata for reproducibility."""

    evalscope_version: str
    model_id: str
    api_url: str
    eval_type: str
    dataset_hub: Literal['modelscope', 'huggingface']
    benchmarks: List[str]


class AgentEvaluationResult(AgentApiModel):
    """Structured result returned to the unified FastAPI task lifecycle."""

    task_type: Literal['agent'] = 'agent'
    summary: Dict[str, Any]
    artifacts: AgentEvaluationArtifacts
    provenance: AgentEvaluationProvenance
