import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GUIDES = sorted((ROOT / "docs" / "guides").glob("[0-9][0-9]-*.md"))
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)


def _lines(text: str) -> list[str]:
    """Lines without indentation or blank lines, so a dedented excerpt still matches."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _contains(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def test_there_is_one_guide_per_example():
    examples = sorted(p.name[:2] for p in (ROOT / "examples").glob("[0-9][0-9]_*.py"))
    assert [g.name[:2] for g in GUIDES] == examples


@pytest.mark.parametrize("guide", GUIDES, ids=lambda g: g.name)
def test_python_blocks_are_verbatim_excerpts_of_the_example(guide):
    source = _lines(next((ROOT / "examples").glob(f"{guide.name[:2]}_*.py")).read_text())
    blocks = [code for lang, code in FENCE.findall(guide.read_text()) if lang == "python"]
    assert blocks, f"{guide.name} has no python block"
    for code in blocks:
        assert _contains(source, _lines(code)), f"{guide.name}: block is not in the example:\n{code}"


@pytest.mark.parametrize("guide", GUIDES, ids=lambda g: g.name)
def test_guide_has_no_dashes_and_its_links_resolve(guide):
    text = guide.read_text()
    assert "—" not in text and "–" not in text, "em or en dash"
    assert text.startswith(f"# Example {guide.name[:2]}:")
    for target in re.findall(r"\]\((?!http)([^)#]+)(?:#[^)]*)?\)", text):
        assert (guide.parent / target).resolve().exists(), f"{guide.name}: broken link {target}"
