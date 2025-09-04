from typing import Any, Dict

from opik.evaluation.metrics import LevenshteinRatio
from opik.evaluation.metrics.score_result import ScoreResult

from opik_optimizer import TaskConfig, datasets
from opik_optimizer.gepa_optimizer import GepaOptimizer


def test_gepa_optimizer() -> None:
    # Optimizer under test
    optimizer = GepaOptimizer(
        model="openai/gpt-4o-mini",
        reflection_model="openai/gpt-4o-mini",
        project_name="gepa_optimization_project",
        temperature=0.1,
        max_tokens=1000,
    )

    # Dataset
    dataset = datasets.tiny_test()

    # Metric
    def levenshtein_ratio(dataset_item: Dict[str, Any], llm_output: str) -> ScoreResult:
        return LevenshteinRatio().score(
            reference=dataset_item["label"], output=llm_output
        )

    # Task config for tiny_test
    task_config = TaskConfig(
        instruction_prompt="Answer with the exact expected label only.",
        input_dataset_fields=["text"],
        output_dataset_field="label",
        use_chat_prompt=True,
        tools=[],
    )

    # Run GEPA with small budget
    results = optimizer.optimize_prompt(
        dataset=dataset,
        metric=levenshtein_ratio,
        task_config=task_config,
        max_metric_calls=6,
        reflection_minibatch_size=2,
        n_samples=3,
    )

    # Core validation
    assert results.optimizer == "GepaOptimizer"
    assert isinstance(results.score, (int, float))
    assert 0.0 <= results.score <= 1.0
    assert results.metric_name == "levenshtein_ratio"

    # Prompt structure
    assert isinstance(results.prompt, list) and len(results.prompt) > 0
    for msg in results.prompt:
        assert isinstance(msg, dict)
        assert "role" in msg and "content" in msg
        assert msg["role"] in ["system", "user", "assistant"]
        assert isinstance(msg["content"], str)

    # Details and history
    assert isinstance(results.details, dict)
    assert isinstance(results.history, list)

    # LLM calls optional
    if results.llm_calls is not None:
        assert isinstance(results.llm_calls, int)
        assert results.llm_calls >= 0

    # Check str() and model_dump()
    s = str(results)
    assert isinstance(s, str)
    d = results.model_dump()
    for key in [
        "optimizer",
        "score",
        "metric_name",
        "prompt",
        "initial_prompt",
        "initial_score",
        "details",
        "history",
    ]:
        assert key in d


if __name__ == "__main__":
    test_gepa_optimizer()

