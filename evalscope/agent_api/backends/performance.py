"""Performance benchmark execution without HTTP or task-lifecycle dependencies."""

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from pydantic import BaseModel, SecretStr

from evalscope import __version__
from evalscope.agent_api.schemas.performance import (
    PerformanceArtifacts,
    PerformanceProvenance,
    PerformanceTestConfig,
    PerformanceTestResult,
)
from evalscope.perf.arguments import Arguments
from evalscope.utils.uri_utils import validate_http_endpoint

CredentialResolver = Callable[[str], str | SecretStr]


def _run_perf_benchmark(arguments: Arguments) -> Any:
    """Import the optional Perf runtime only when a benchmark is executed."""

    from evalscope.perf.main import run_perf_benchmark

    return run_perf_benchmark(arguments)


def _serialize(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return '<redacted>'
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json', by_alias=True, exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'to_dict'):
        return _serialize(value.to_dict())
    return str(value)


def build_performance_arguments(
    config: PerformanceTestConfig,
    credential_resolver: CredentialResolver,
) -> Arguments:
    """Resolve credentials and build EvalScope performance arguments."""

    validate_http_endpoint(config.target.api_url)
    value = credential_resolver(config.target.credential_ref)
    api_key = value if isinstance(value, SecretStr) else SecretStr(value)
    if not api_key.get_secret_value():
        raise ValueError('target credential resolved to an empty value')

    return Arguments(
        model=config.target.model_id,
        api=config.target.api,
        url=config.target.api_url,
        api_key=api_key,
        tokenizer_path=config.target.tokenizer_path,
        dataset=config.dataset,
        dataset_path=config.dataset_path,
        dataset_args=config.dataset_args,
        number=config.number,
        parallel=config.parallel,
        rate=config.rate,
        open_loop=config.open_loop,
        duration=config.duration,
        warmup_num=config.warmup_num,
        stream=config.stream,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        min_prompt_length=config.min_prompt_length,
        max_prompt_length=config.max_prompt_length,
        prompt=config.prompt,
        temperature=config.temperature,
        outputs_dir=config.output_dir,
        no_timestamp=config.no_timestamp,
        no_test_connection=config.no_test_connection,
        seed=config.seed,
    )


def run_performance_test(
    config: PerformanceTestConfig,
    credential_resolver: CredentialResolver,
) -> PerformanceTestResult:
    """Run EvalScope Perf and return a transport-neutral, secret-free result."""

    arguments = build_performance_arguments(config, credential_resolver)
    summary = _serialize(_run_perf_benchmark(arguments))
    return PerformanceTestResult(
        summary=summary,
        artifacts=PerformanceArtifacts(output_dir=arguments.outputs_dir),
        provenance=PerformanceProvenance(
            evalscope_version=__version__,
            model_id=config.target.model_id,
            api_url=config.target.api_url,
            api=config.target.api,
            dataset=config.dataset,
        ),
    )
