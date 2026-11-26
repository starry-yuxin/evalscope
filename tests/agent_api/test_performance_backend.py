import json
from dataclasses import dataclass

import pytest
from pydantic import SecretStr, ValidationError

from evalscope.agent_api.backends.performance import build_performance_arguments, run_performance_test
from evalscope.agent_api.schemas.performance import PerformanceTarget, PerformanceTestConfig


def _target():
    return PerformanceTarget(
        model_id='Qwen/Qwen2.5-0.5B-Instruct',
        api_url='http://127.0.0.1:8801/v1',
        credential_ref='local-model-key',
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('number', 0),
        ('number', 1001),
        ('parallel', 0),
        ('parallel', 33),
        ('duration', 1801),
        ('max_tokens', 2049),
        ('max_prompt_length', 16385),
        ('temperature', 2.1),
    ],
)
def test_performance_config_enforces_safety_bounds(field, value):
    with pytest.raises(ValidationError):
        PerformanceTestConfig(target=_target(), **{field: value})


def test_performance_config_validates_sweep_shapes_and_token_ranges():
    with pytest.raises(ValidationError, match='number and parallel'):
        PerformanceTestConfig(target=_target(), number=[10, 20], parallel=[1])

    with pytest.raises(ValidationError, match='number and rate'):
        PerformanceTestConfig(target=_target(), number=[10, 20], rate=[1], open_loop=True)

    with pytest.raises(ValidationError, match='open-loop rate'):
        PerformanceTestConfig(target=_target(), number=10, rate=-1, open_loop=True)

    with pytest.raises(ValidationError, match='min_tokens'):
        PerformanceTestConfig(target=_target(), min_tokens=65, max_tokens=64)

    with pytest.raises(ValidationError, match='min_prompt_length'):
        PerformanceTestConfig(target=_target(), min_prompt_length=100, max_prompt_length=50)


def test_build_performance_arguments_maps_values_and_secret(tmp_path):
    config = PerformanceTestConfig(
        target=_target(),
        dataset='random',
        number=[10, 20],
        parallel=[1, 2],
        max_tokens=64,
        max_prompt_length=128,
        output_dir=str(tmp_path),
        no_test_connection=True,
    )

    arguments = build_performance_arguments(config, lambda ref: f'secret-for-{ref}')

    assert arguments.model == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert arguments.url.endswith('/v1/chat/completions')
    assert arguments.number == [10, 20]
    assert arguments.parallel == [1, 2]
    assert arguments.max_tokens == 64
    assert arguments.max_prompt_length == 128
    assert arguments.outputs_dir == str(tmp_path)
    assert arguments.api_key.get_secret_value() == 'secret-for-local-model-key'
    assert arguments.headers['Authorization'].get_secret_value() == 'Bearer secret-for-local-model-key'


@dataclass
class _FakeMetrics:
    request_throughput: float
    average_latency: float


def test_run_performance_test_serializes_result_and_excludes_secret(monkeypatch, tmp_path):
    def fake_run(arguments):
        arguments.outputs_dir = str(tmp_path / '20261126_120000' / 'Qwen2.5-0.5B-Instruct')
        return {
            'parallel_2_number_4': {
                'metrics': _FakeMetrics(request_throughput=8.5, average_latency=0.25),
                'percentiles': {'p99': 0.5},
                'secret_check': SecretStr('should-never-leak'),
            }
        }

    monkeypatch.setattr('evalscope.agent_api.backends.performance._run_perf_benchmark', fake_run)
    config = PerformanceTestConfig(
        target=_target(),
        number=4,
        parallel=2,
        output_dir=str(tmp_path),
        no_test_connection=True,
    )

    result = run_performance_test(config, lambda _: 'resolved-api-secret')
    payload = result.model_dump(mode='json')
    encoded = json.dumps(payload)
    run = payload['summary']['parallel_2_number_4']

    assert payload['task_type'] == 'performance'
    assert run['metrics']['request_throughput'] == 8.5
    assert run['percentiles']['p99'] == 0.5
    assert run['secret_check'] == '<redacted>'
    assert payload['artifacts']['output_dir'].endswith('Qwen2.5-0.5B-Instruct')
    assert payload['provenance']['dataset'] == 'openqa'
    assert 'resolved-api-secret' not in encoded
    assert 'should-never-leak' not in encoded
