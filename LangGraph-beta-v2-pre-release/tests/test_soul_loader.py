import os
import tempfile

from src.core.local_llm import LocalLLMClient, LocalLLMConfig
from src.core.soul_loader import (
    SoulLoader,
    clear_soul_cache,
    list_available_souls,
    load_soul,
    normalize_role_name,
)


def test_normalize_role_name():
    assert normalize_role_name("Researcher") == "researcher"
    assert normalize_role_name("coder_soul") == "coder"
    assert normalize_role_name("critic.md") == "critic"
    assert normalize_role_name("  writer_soul.md  ") == "writer"


def test_default_soul_loader_real_souls():
    clear_soul_cache()
    # Load researcher soul
    researcher_soul = load_soul("researcher")
    assert len(researcher_soul) > 0
    assert "researcher" in researcher_soul.lower()

    # Verify cached loading
    cached_soul = load_soul("RESEARCHER")
    assert cached_soul == researcher_soul


def test_soul_loader_fallback():
    fallback = "Default persona rules"
    result = load_soul("non_existent_role_xyz", fallback_prompt=fallback)
    assert result == fallback


def test_custom_souls_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        soul_file = os.path.join(tmp_dir, "custom_agent_soul.md")
        with open(soul_file, "w", encoding="utf-8") as f:
            f.write("You are a custom agent persona.")

        loader = SoulLoader(souls_dir=tmp_dir)
        content = loader.load_soul("custom_agent")
        assert content == "You are a custom agent persona."

        available = loader.list_available_souls()
        assert available == ["custom_agent"]


def test_reload_and_clear_cache():
    with tempfile.TemporaryDirectory() as tmp_dir:
        soul_file = os.path.join(tmp_dir, "dynamic_soul.md")
        with open(soul_file, "w", encoding="utf-8") as f:
            f.write("Version 1")

        loader = SoulLoader(souls_dir=tmp_dir)
        assert loader.load_soul("dynamic") == "Version 1"

        # Update file on disk
        with open(soul_file, "w", encoding="utf-8") as f:
            f.write("Version 2")

        # Before reload, cache returns Version 1
        assert loader.load_soul("dynamic") == "Version 1"

        # With reload=True, fresh content is read
        assert loader.load_soul("dynamic", reload=True) == "Version 2"

        # Update file again
        with open(soul_file, "w", encoding="utf-8") as f:
            f.write("Version 3")

        # Clearing cache forces read of Version 3
        loader.clear_cache("dynamic")
        assert loader.load_soul("dynamic") == "Version 3"


def test_format_soul():
    with tempfile.TemporaryDirectory() as tmp_dir:
        soul_file = os.path.join(tmp_dir, "template_soul.md")
        with open(soul_file, "w", encoding="utf-8") as f:
            f.write("Hello $name, your role is $role.")

        loader = SoulLoader(souls_dir=tmp_dir)
        formatted = loader.format_soul("template", name="Alice", role="Supervisor")
        assert formatted == "Hello Alice, your role is Supervisor."


def test_list_available_souls():
    available = list_available_souls()
    assert isinstance(available, list)
    assert "researcher" in available
    assert "coder" in available


def test_local_llm_agent_model_selection():
    config = LocalLLMConfig(
        model_name="default-model",
        agent_models={"coder": "coder-model", "critic": "critic-model"},
    )
    client = LocalLLMClient(config=config)

    assert client._resolve_model_name(agent="coder") == "coder-model"
    assert client._resolve_model_name(agent="critic") == "critic-model"
    assert client._resolve_model_name(agent="unknown") == "default-model"
    assert client._resolve_model_name(model_name="override") == "override"


def test_build_search_context_preserves_more_research_text():
    client = LocalLLMClient(config=LocalLLMConfig(provider="mock"))
    long_context = "A" * 6000

    context = client.build_search_context(long_context, max_chars=8000)

    assert context == long_context
    assert len(context) > 5000
