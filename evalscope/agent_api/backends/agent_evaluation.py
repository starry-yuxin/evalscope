"""Agent benchmark execution without HTTP or task-lifecycle dependencies."""

import os
import re
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, SecretStr

from evalscope import TaskConfig, __version__, run_task
from evalscope.agent_api.schemas.agent import (
    AgentEvaluationArtifacts,
    AgentEvaluationConfig,
    AgentEvaluationProvenance,
    AgentEvaluationResult,
)
from evalscope.utils.resource_utils import load_benchmark_data
from evalscope.utils.uri_utils import validate_http_endpoint

CredentialResolver = Callable[[str], str | SecretStr]


class AgentEvaluationValidationError(ValueError):
    """Raised when an Agent evaluation cannot be mapped safely to EvalScope."""


def _safe_name(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', value).strip('-') or 'benchmark'


def _serialize(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return '<redacted>'
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'to_dict'):
        return _serialize(value.to_dict())
    return str(value)


def validate_agent_evaluation(config: AgentEvaluationConfig) -> Dict[str, Dict[str, Any]]:
    """Validate endpoint, Agent category, and selected benchmark subsets."""

    try:
        validate_http_endpoint(config.target.api_url)
    except ValueError as exc:
        raise AgentEvaluationValidationError(str(exc)) from exc

    metadata = {}
    for benchmark_id in config.benchmarks:
        entry = load_benchmark_data(benchmark_id).get(benchmark_id, {})
        meta = entry.get('meta') or {}
        if not meta:
            raise AgentEvaluationValidationError(f'benchmark {benchmark_id!r} was not found')
        if meta.get('category') != 'agent':
            raise AgentEvaluationValidationError(f'benchmark {benchmark_id!r} is not an Agent benchmark')

        options = config.benchmark_options.get(benchmark_id)
        selected_subsets = options.subsets if options else None
        if selected_subsets is not None:
            available = list(meta.get('subset_list') or [])
            unknown = sorted(set(selected_subsets) - set(available))
            if unknown:
                raise AgentEvaluationValidationError(
                    f'unknown subsets for benchmark {benchmark_id!r}: {", ".join(unknown)}; '
                    f'available subsets: {", ".join(available)}'
                )
        metadata[benchmark_id] = meta
    return metadata


def _resolve_api_key(config: AgentEvaluationConfig, resolver: CredentialResolver) -> SecretStr:
    value = resolver(config.target.credential_ref)
    secret = value if isinstance(value, SecretStr) else SecretStr(value)
    if not secret.get_secret_value():
        raise AgentEvaluationValidationError('target credential resolved to an empty value')
    return secret


def build_agent_task_configs(
    config: AgentEvaluationConfig,
    credential_resolver: CredentialResolver,
) -> List[TaskConfig]:
    """Build one native TaskConfig per Agent benchmark in request order."""

    validate_agent_evaluation(config)
    api_key = _resolve_api_key(config, credential_resolver)
    task_configs = []
    for benchmark_id in config.benchmarks:
        options = config.benchmark_options.get(benchmark_id)
        dataset_args = dict(options.dataset_args) if options else {}
        if options and options.subsets is not None:
            dataset_args['subset_list'] = list(options.subsets)
        limit = (
            options.limit_per_subset if options and options.limit_per_subset is not None else config.limit_per_subset
        )
        task_configs.append(
            TaskConfig(
                model=config.target.model_id,
                model_id=config.target.model_id,
                api_url=config.target.api_url,
                api_key=api_key,
                eval_type=config.target.eval_type,
                datasets=[benchmark_id],
                dataset_args={benchmark_id: dataset_args} if dataset_args else {},
                dataset_hub=config.dataset_hub,
                limit=limit,
                repeats=config.repeats,
                generation_config=config.generation.model_dump(exclude_none=True),
                eval_batch_size=config.eval_batch_size,
                ignore_errors=config.ignore_errors,
                seed=config.seed,
                work_dir=os.path.join(config.work_dir, _safe_name(benchmark_id)),
                no_timestamp=True,
                use_cache=config.use_cache,
                rerun_review=config.rerun_review,
            )
        )
    return task_configs


def run_agent_evaluation(
    config: AgentEvaluationConfig,
    credential_resolver: CredentialResolver,
) -> AgentEvaluationResult:
    """Run Agent benchmarks through EvalScope and return a transport-neutral result."""

    task_configs = build_agent_task_configs(config, credential_resolver)
    summary = {}
    benchmark_roots = {}
    for task_config in task_configs:
        benchmark_id = task_config.datasets[0]
        summary[benchmark_id] = _serialize(run_task(task_config))
        benchmark_roots[benchmark_id] = task_config.work_dir

    return AgentEvaluationResult(
        summary=summary,
        artifacts=AgentEvaluationArtifacts(root=config.work_dir, benchmark_roots=benchmark_roots),
        provenance=AgentEvaluationProvenance(
            evalscope_version=__version__,
            model_id=config.target.model_id,
            api_url=config.target.api_url,
            eval_type=config.target.eval_type,
            dataset_hub=config.dataset_hub,
            benchmarks=list(config.benchmarks),
        ),
    )
