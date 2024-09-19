import pytest
from src.llm_wrapper import LLMWrapper
from giskard import testing, Dataset, LLMUse

@pytest.fixture
def llm_wrapper():
    return LLMWrapper()

def test_generate_response(llm_wrapper):
    response = llm_wrapper.generate_response("What's a good investment strategy?")
    assert len(response) > 0
    assert "investment" in response.lower() or "strategy" in response.lower()

def test_llm_performance():
    dataset = Dataset(
        [
            "What's a good investment strategy?",
            "How do I save for retirement?",
            "Should I invest in stocks or bonds?",
        ],
        name="financial_queries"
    )

    llm_use = LLMUse(
        handle_response=lambda x: LLMWrapper().generate_response(x),
        usage_examples=[("What's a good investment strategy?", "Here's a balanced investment strategy...")]
    )

    results = testing.test_suite(
        llm_use,
        dataset,
        [
            testing.test_accuracy(),
            testing.test_toxicity(),
            testing.test_polarity(),
            testing.test_named_entity_recognition(),
        ]
    )

    assert results.passed()
