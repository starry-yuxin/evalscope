import json

import pytest
from pydantic import SecretStr, ValidationError

from evalscope.agent_api.backends.agent_evaluation import (
    AgentEvaluationValidationError,
    build_agent_task_configs,
    run_agent_evaluation,
    validate_agent_evaluation,
)
from evalscope.agent_api.contracts import ModelTarget
from evalscope.agent_api.schemas.agent import AgentEvaluationConfig


def _target():
    return ModelTarget(
        model_id='test-model',
        api_url='https://model.example.test/v1',
        credential_ref='target-key',
    )


def test_agent_config_rejects_duplicates_and_invalid_limits():
    with pytest.raises(ValidationError, match='benchmarks must not contain duplicates'):
        AgentEvaluationConfig(target=_target(), benchmarks=['general_fc', 'general_fc'])

    with pytest.raises(ValidationError, match='limit_per_subset'):
        AgentEvaluationConfig(
            target=_target(),
            benchmarks=['general_fc'],
            benchmark_options={'general_fc': {'limit_per_subset': 1.5}},
        )

    with pytest.raises(ValidationError, match='subsets must not contain duplicates'):
        AgentEvaluationConfig(
            target=_target(),
            benchmarks=['general_fc'],
            benchmark_options={'general_fc': {'subsets': ['default', 'default']}},
        )


def test_agent_validation_rejects_non_agent_benchmark_and_unknown_subset():
    with pytest.raises(AgentEvaluationValidationError, match='not an Agent benchmark'):
        validate_agent_evaluation(AgentEvaluationConfig(target=_target(), benchmarks=['gsm8k']))

    config = AgentEvaluationConfig(
        target=_target(),
        benchmarks=['general_fc'],
        benchmark_options={'general_fc': {'subsets': ['missing']}},
    )
    with pytest.raises(AgentEvaluationValidationError, match='unknown subsets'):
        validate_agent_evaluation(config)


def test_build_agent_task_config_uses_native_contract_and_secret(tmp_path):
    config = AgentEvaluationConfig(
        target=_target(),
        benchmarks=['general_fc'],
        benchmark_options={
            'general_fc': {
                'subsets': ['default'],
                'limit_per_subset': 5,
                'dataset_args': {'custom_flag': True},
            }
        },
        generation={'temperature': 0.2, 'max_tokens': 64},
        eval_batch_size=2,
        repeats=2,
        work_dir=str(tmp_path),
    )

    task_config = build_agent_task_configs(config, lambda ref: f'secret-for-{ref}')[0]

    assert task_config.datasets == ['general_fc']
    assert task_config.dataset_args == {
        'general_fc': {'custom_flag': True, 'subset_list': ['default']},
    }
    assert task_config.limit == 5
    assert task_config.eval_batch_size == 2
    assert task_config.repeats == 2
    assert task_config.generation_config.max_tokens == 64
    assert task_config.api_key.get_secret_value() == 'secret-for-target-key'
    assert task_config.no_timestamp is True
    assert task_config.work_dir.endswith('general_fc')


def test_run_agent_evaluation_returns_structured_secret_free_result(monkeypatch, tmp_path):
    captured = []

    def fake_run_task(task_config):
        captured.append(task_config)
        return {'score': 0.75, 'secret_check': SecretStr('should-never-leak')}

    monkeypatch.setattr('evalscope.agent_api.backends.agent_evaluation.run_task', fake_run_task)
    config = AgentEvaluationConfig(
        target=_target(),
        benchmarks=['general_fc'],
        limit_per_subset=1,
        work_dir=str(tmp_path),
    )

    result = run_agent_evaluation(config, lambda _: 'resolved-api-secret')
    payload = result.model_dump(mode='json')
    encoded = json.dumps(payload)

    assert len(captured) == 1
    assert payload['task_type'] == 'agent'
    assert payload['summary']['general_fc']['score'] == 0.75
    assert payload['summary']['general_fc']['secret_check'] == '<redacted>'
    assert payload['artifacts']['root'] == str(tmp_path)
    assert payload['provenance']['benchmarks'] == ['general_fc']
    assert 'resolved-api-secret' not in encoded
    assert 'should-never-leak' not in encoded
