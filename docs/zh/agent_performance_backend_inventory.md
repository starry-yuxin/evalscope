# Agent 评测与性能压测能力梳理

> 编制日期：2026-11-26  
> 目标分支：`feat/agent-performance-backend`  
> 整合目标：将已验证的 Agent 评测和性能压测能力拆成独立 Python 后端，由统一 FastAPI 调用。

## 1. 范围与结论

当前工作建立在 EvalScope 原生评测和 Perf 压测能力之上，没有重新实现 Benchmark 数据加载、模型调用、评分或性能指标计算。现有新增代码主要提供受控配置、异步任务、SQLite 持久化和 Flask HTTP 接口。

本次整合保留以下能力：

- Agent Benchmark 的配置构建和 EvalScope 调用。
- EvalScope Perf 的配置构建、参数边界和结构化结果。
- 模型、数据集和压测场景的受控注册信息。
- 已验证实验的配置、指标口径和最小运行示例。

本次整合不直接迁移以下组件：

- Flask Blueprint、`request`、`jsonify` 和 Flask 全局应用对象。
- 原有线程池任务调度器。
- SQLite Job Store 和第二套任务状态机。
- 原接口中的任务提交、状态查询、结果查询和取消路由。

统一 FastAPI 继续负责鉴权、任务生命周期、事件、进度、取消和结果查询；新增后端仅接收普通 Python/Pydantic 配置并返回结构化结果。

## 2. Agent 评测能力

### 2.1 已完成真实评测的 Benchmark

| Benchmark | 数据规模 | 已验证范围 | 主要指标 | 当前状态 |
| --- | ---: | --- | --- | --- |
| `general_fc` | 2,000 条，1 个子集 | 50 条 | Tool-call F1、Schema Accuracy | 已完成真实 API 评测 |
| `bfcl_v4` | 5,106 条，22 个子集 | 4 个单轮子集各 50 条，共 200 条；另有 1 条多轮样本 | Accuracy | 已完成真实 API 评测 |
| `gsm8k` + DeepSeek Harness | GSM8K `main` | 1 条 | Accuracy | 已完成 Harness/Docker 链路冒烟 |

这些结果用于验证执行链路和指标解析，不代表模型在完整数据集上的正式成绩。

### 2.2 `general_fc`

`general_fc` 用于评估通用函数调用和工具选择能力。当前验证配置如下：

| 配置项 | 值 |
| --- | --- |
| 模型 | `deepseek-flash` |
| 模型模式 | non-thinking |
| 样本数 | 50 |
| 随机种子 | 42 |
| 生成温度 | 0 |
| 主要指标 | Tool-call F1、Schema Accuracy |

验证结果：

| 指标 | 结果 |
| --- | ---: |
| 成功样本 / API 错误 | 50 / 0 |
| Tool-call F1 | 68.2% |
| Schema Accuracy | 77.8%（21/27） |
| 工具调用召回率 | 88.2%（15/17） |
| 工具调用精确率 | 55.6%（15/27） |

主要失败来自不必要工具调用、调用未提供的工具和少量漏调。当前统一接口的 YAML 清单仅登记了一个最小配置：

```yaml
general_fc_5_v1:
  datasets:
    - general_fc
  limit: 5
  eval_batch_size: 4
```

因此需要区分：`general_fc` 的 50 条实验已经验证，但原 Flask 接口清单目前只直接暴露 `general_fc_5_v1`。

最小 EvalScope 配置示意：

```python
from evalscope import TaskConfig, run_task

config = TaskConfig(
    model='deepseek-flash',
    model_id='deepseek-flash-nonthinking',
    eval_type='openai_api',
    api_url='https://api.deepseek.com/v1',
    datasets=['general_fc'],
    limit=5,
    eval_batch_size=4,
    generation_config={
        'temperature': 0,
        'extra_body': {'thinking': {'type': 'disabled'}},
    },
)
run_task(config)
```

API Key 必须由服务端凭据解析器注入，不进入请求、注册文件或日志。

### 2.3 `bfcl_v4`

BFCL-v4 用于评估函数选择、参数构造、并行函数调用、无关问题拒绝和多轮工具调用。当前扩大验证覆盖以下子集：

- `simple_python`
- `multiple`
- `parallel`
- `irrelevance`

200 条单轮样本结果：

| 子集 | 正确 / 样本 | Accuracy |
| --- | ---: | ---: |
| `irrelevance` | 39 / 50 | 78% |
| `multiple` | 43 / 50 | 86% |
| `parallel` | 44 / 50 | 88% |
| `simple_python` | 46 / 50 | 92% |
| 合计 | 172 / 200 | 86% |

另使用 `multi_turn_base` 完成 1 条真实多轮样本，24 次模型请求均返回 HTTP 200，最终 Accuracy 为 100%。单样本结果仅证明 DeepSeek thinking、多轮历史和工具结果回传链路可运行。

BFCL-v4 运行依赖：

- 独立 Python 环境。
- `bfcl-eval==2025.10.27.1`。
- 当前环境还需要显式安装 `soundfile==0.14.0`。
- 多轮样本可能产生大量模型调用和输入 Token，扩大前必须估算成本。

需要注意，部分子集运行时 BFCL 报告中的 `OVERALL` 会把未运行类别按零计入完整权重。局部实验应使用所选样本的 `accuracy`，不能把该 `OVERALL` 当成本次抽样准确率。

当前 BFCL-v4 尚未登记到原 Flask 接口的 `datasets.yaml`，属于已经实跑验证、等待接入统一注册信息的能力。

### 2.4 DeepSeek Harness + GSM8K

该实验验证了以下链路：

```text
EvalScope -> DeepSeek Harness -> Docker -> EvalScope Model Proxy -> DeepSeek API
```

当前服务器报告记录的是 1 条 GSM8K `main` 样本冒烟：

| 指标 | 结果 |
| --- | ---: |
| 请求样本 | 1 |
| 成功样本 | 1 |
| Accuracy | 100%（1/1） |
| Harness 超时 | 否 |
| 实际模型请求 | 2 |

该结果证明 Harness、Docker、答案提取和评分链路可运行，不具备统计代表性。Harness 的性能数据包含辅助模型请求，不能等同于一次裸模型问答。

### 2.5 已盘点但尚未完成统一验证的 Benchmark

仓库还包含 GAIA、SWE-bench Agentic、Tau Bench、MiniWoB、DeepSearchQA、Terminal-Bench 等 Agent Benchmark。它们可能依赖 Docker、浏览器、Judge、外部工具或相互冲突的 Python 依赖。

这些项目目前属于能力盘点，不应在 API 能力目录中标记为 `verified`，也不应宣称已完成端到端支持。

## 3. 性能压测能力

### 3.1 已验证目标和场景

本地 `Qwen2.5-0.5B-Instruct` 通过 OpenAI Chat Completions 兼容服务运行：

| 配置项 | 值 |
| --- | --- |
| 模型引用 | `local-qwen2.5-0.5b` |
| 模型服务 | `http://127.0.0.1:8801/v1` |
| 模型权重 | `/home/evalagent/zhq/models/Qwen2.5-0.5B-Instruct` |
| 压测后端 | EvalScope Perf |
| 数据集 | `openqa` |
| 流式响应 | 是 |
| 最大输出 | 128 Token |

已登记并验证两个主要场景：

| 场景 | 请求数 | 并发 | 预热 | 用途 |
| --- | ---: | --- | ---: | --- |
| `local_qwen_smoke_10_v1` | 10 | 1 | 0 | 接口和模型服务冒烟 |
| `local_qwen_capacity_192_v1` | 64 x 3 | 8、16、32 | 每档 8 | 容量和延迟梯度 |

容量测试结果：

| 并发 | 成功/失败 | 请求吞吐 | 输出吞吐 | 平均延迟 | 平均 TTFT | 平均 TPOT | P95 延迟 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64/0 | 7.80 req/s | 867.31 tok/s | 0.917 s | 19.08 ms | 8.14 ms | 1.078 s |
| 16 | 64/0 | 11.11 req/s | 1232.26 tok/s | 1.143 s | 30.29 ms | 10.11 ms | 1.359 s |
| 32 | 64/0 | 14.81 req/s | 1636.31 tok/s | 1.619 s | 65.01 ms | 14.12 ms | 1.960 s |

192 个计量请求全部成功。以上结果仅描述当时机器、模型服务和测试配置，不应脱离环境信息进行横向比较。

### 3.2 当前封装支持的输入参数

| 参数 | 作用 | 当前约束 |
| --- | --- | --- |
| `dataset` | Perf 数据集插件 | 场景中登记 |
| `dataset_path` | 自定义数据路径 | 场景中登记，不由远程请求直接传入 |
| `number` | 每个测试点的请求数 | 每项 1-1000 |
| `parallel` | 闭环并发数 | 每项 1-32 |
| `rate` | 目标请求速率 | 开环模式每项大于 0 且不超过 100 |
| `open_loop` | 使用开环到达率 | 场景中登记 |
| `duration` | 测试持续时间 | 大于 0 且不超过 1800 秒 |
| `warmup_num` | 预热请求数或比例 | 场景中登记 |
| `stream` | 是否流式响应 | 可受限覆盖 |
| `max_tokens` | 最大输出 Token | 1-2048 |
| `min_tokens` | 最小输出 Token | 0 到 `max_tokens` |
| `min_prompt_length` | 最小输入长度 | 不小于 0 |
| `max_prompt_length` | 最大输入长度 | 不超过 16384 |
| `prompt` | 固定测试 Prompt | 场景中登记，不由远程请求直接传入 |
| `tokenizer_path` | Tokenizer 路径 | 服务端场景配置 |
| `temperature` | 生成温度 | 0-2 |

原 Flask HTTP 请求只允许覆盖 `number`、`parallel`、`rate`、`duration` 和 `stream`。其余字段必须由经过审核的服务端场景提供，从而避免调用方指定任意目标、任意本地路径或无边界负载。

EvalScope Perf 本身还支持开环压测、多点扫描、固定时长、SLA 自动调优、多轮对话、视觉多模态和生产流量回放；这些是 EvalScope 原生能力，不等于当前 Agent 接口已经全部开放。

### 3.3 输出指标

当前结果直接保留 EvalScope Perf 的结构化汇总和分位数，主要包括：

- 总请求数、成功请求数、失败请求数和成功率。
- 测试持续时间和请求吞吐量（RPS/QPS）。
- 输入、输出和总 Token 吞吐量。
- 端到端延迟（Latency）。
- 首 Token 延迟（TTFT）。
- 每输出 Token 时间（TPOT）。
- Token/Chunk 间延迟（ITL）。
- Avg、P50、P90、P95、P99 和 Max 等统计。
- 适用场景下的多轮缓存、首轮/后续轮 TTFT 和 PD 分离指标。
- 结果目录、HTML 报告、原始数据库和日志路径。

## 4. EvalScope 原有能力与新增能力边界

| 能力 | 归属 | 说明 |
| --- | --- | --- |
| `TaskConfig`、`run_task` | EvalScope 原有 | 质量评测主入口 |
| Benchmark Adapter 和注册机制 | EvalScope 原有 | 数据加载、样本转换、模型输入和评分 |
| Agent Benchmark 数据与指标 | EvalScope 原有 | `general_fc`、BFCL-v4、GSM8K 等 |
| `PerfArguments`、`run_perf_benchmark` | EvalScope 原有 | 性能压测执行入口 |
| Perf 请求生成、调度和指标计算 | EvalScope 原有 | RPS、Token 吞吐、延迟和分位数 |
| EvalScope 输出目录和原生报告 | EvalScope 原有 | 日志、预测、评分、报告和进度 |
| 模型/数据集/场景 YAML 清单 | 当前新增 | 对可调用资源进行别名登记和限制 |
| 清单 Pydantic 校验和安全上限 | 当前新增 | 校验引用、参数边界和覆盖规则 |
| `TaskConfig`/`PerfArguments` 构建 | 当前新增 | 从受控注册信息构造 EvalScope 原生配置 |
| 结构化结果和 provenance | 当前新增 | 保存版本、提交、别名、场景和产物路径 |
| Flask Blueprint 路由 | 当前新增 | 原 Agent HTTP 接口，整合时不迁移 |
| SQLite Job Store | 当前新增 | 原任务持久化，整合时先独立评估 |
| 线程池 Runner 和取消逻辑 | 当前新增 | 原任务生命周期，整合时由 FastAPI 替代 |

## 5. 当前完整调用流程

### 5.1 Agent/质量评测

```text
POST /v1/evaluations
  -> Flask/Pydantic 校验请求
  -> 根据 model_ref 和 dataset_ref 查询 YAML 清单
  -> 校验 API Key 环境变量和 limit 覆盖
  -> 构造 EvalScope TaskConfig
  -> SQLite 创建 queued 作业
  -> 线程池启动任务
  -> 子进程执行 run_task
  -> EvalScope 生成日志、预测、评分和报告
  -> 序列化结果并写入 SQLite
  -> 调用方查询状态、结果或日志
```

### 5.2 性能压测

```text
POST /v1/load-tests
  -> Flask/Pydantic 校验请求
  -> 根据 model_ref 和 scenario_ref 查询 YAML 清单
  -> 检查模型是否明确允许压测
  -> 校验受限覆盖和安全上限
  -> 构造 EvalScope PerfArguments
  -> SQLite 创建 queued 作业
  -> 线程池启动任务
  -> 子进程执行 run_perf_benchmark
  -> EvalScope 生成指标、分位数、数据库、日志和 HTML 报告
  -> 序列化结果并写入 SQLite
  -> 调用方查询状态、结果或日志
```

## 6. 当前输入与输出格式

### 6.1 Agent 评测输入

```json
{
  "idempotency_key": "general-fc-smoke-001",
  "task_type": "evaluation",
  "model_ref": "deepseek-flash-nonthinking",
  "dataset_ref": "general_fc_5_v1",
  "limit": 5,
  "generation_config": {},
  "tags": {
    "purpose": "agent-evaluation"
  }
}
```

### 6.2 性能压测输入

```json
{
  "idempotency_key": "local-qwen-capacity-001",
  "task_type": "load_test",
  "model_ref": "local-qwen2.5-0.5b",
  "scenario_ref": "local_qwen_capacity_192_v1",
  "overrides": {
    "parallel": [8, 16, 32],
    "number": [64, 64, 64]
  },
  "tags": {
    "purpose": "capacity-test"
  }
}
```

### 6.3 当前成功结果

质量评测和压测目前使用同一外层结构：

```json
{
  "summary": {},
  "artifacts": {
    "output_dir": "/path/to/output"
  },
  "provenance": {
    "evalscope_version": "...",
    "source_commit": "...",
    "model_ref": "...",
    "tags": {}
  }
}
```

性能压测的 `artifacts` 还包含 `run_dir`，`provenance` 包含 `scenario_ref` 和展开后的场景配置。`summary` 保留 EvalScope 返回对象序列化后的内容。

## 7. SQLite、幂等、重启和日志

### 7.1 SQLite 解决的问题

SQLite Job Store 用于在单机服务中持久保存：

- `job_id`
- 调用方提供的 `idempotency_key`
- 规范化请求的哈希
- 任务类型和状态
- 原始请求 JSON
- 结构化结果 JSON
- 输出目录和安全错误信息
- 创建、开始和结束时间

它解决了历史状态和结果在进程退出后的保留问题，但不负责自动恢复 EvalScope 子进程。

### 7.2 幂等规则

幂等键由调用方提供，不由服务端根据业务字段自动生成。服务端对任务类型和规范化请求计算哈希：

- 相同幂等键和相同请求返回已有 Job。
- 相同幂等键和不同请求返回冲突。
- 数据库唯一约束和立即事务避免并发请求重复创建 Job。

统一 FastAPI 如果引入幂等，应复用这一语义，但不需要复用原 SQLite 状态机。

### 7.3 服务重启行为

服务启动时会把遗留的 `running` 作业标记为 `interrupted`。已完成任务的请求、结果和产物路径仍可查询。当前实现不会自动恢复被中断的子进程，也不会自动重跑任务；排队任务的重建行为也没有形成完整恢复机制。

### 7.4 日志保存和查询

质量评测日志来自：

```text
<output_dir>/logs/eval_log.log
```

性能压测日志来自：

```text
<output_dir>/perf/benchmark.log
```

原 Flask 接口支持按起始行和页大小读取日志。统一 FastAPI 已提供事件文件和结果产物，后续应决定是否继续暴露原始日志分页接口，而不是同时保留两套日志状态。

## 8. 拆分后的后端契约

建议新增以下普通 Python 后端：

```python
def run_agent_evaluation(config: AgentEvaluationConfig) -> AgentEvaluationResult:
    """Build an EvalScope TaskConfig, run it, and return a structured result."""
```

```python
def run_performance_test(config: PerformanceTestConfig) -> PerformanceTestResult:
    """Build EvalScope Perf arguments, run the benchmark, and return metrics."""
```

后端职责限定为：

1. 校验 Agent 或性能压测业务参数。
2. 解析服务端受控模型、数据集和场景信息。
3. 构造 EvalScope 原生配置。
4. 调用 EvalScope 执行入口。
5. 返回结构化结果、产物引用和来源信息。

后端不生成 `job_id`，不访问 Flask 请求，不维护 SQLite 状态，不创建第二套线程池，也不定义 HTTP 响应。取消信号、进度回调、超时和事件应通过统一 FastAPI 的现有任务执行机制传入。

## 9. 建议迁移目录

```text
evalscope/agent_api/
|-- service/                     # 现有统一 FastAPI
|-- backends/
|   |-- __init__.py
|   |-- agent_evaluation.py      # Agent 评测纯 Python 后端
|   `-- performance.py           # Perf 纯 Python 后端
|-- schemas/
|   |-- __init__.py
|   |-- agent.py                 # Agent 业务参数和结果
|   `-- performance.py           # 压测业务参数和结果
`-- tests/
    |-- test_agent_backend.py
    `-- test_performance_backend.py
```

第一阶段只添加 Schema、后端和单元测试，不修改现有 LLM/VLM 调度核心。第二阶段再由统一任务管理增加 `agent` 和 `performance` 类型，并运行 capability、agent、performance 三类任务的提交、状态、结果和取消回归测试。

