"""The documentation site (docs/, a Mintlify site), kept current by tests (SPEC.md M8.5):
every CLI command and option, config key, and MCP tool, prompt and resource has a
reference entry, and pages that quote the code match it. Also: every page is in the
navigation, internal links resolve, and pages are valid MDX as far as plain text can tell."""

import json
import re
from pathlib import Path
from typing import get_args, get_origin

import pytest
import typer.main
from pydantic import BaseModel

from conftest import connect
from openmemore.cli import app, initialize
from openmemore.config import Config

DOCS = Path(__file__).resolve().parent.parent / "docs"


def page(name: str) -> str:
    return (DOCS / f"{name}.mdx").read_text(encoding="utf-8")


def nav_pages() -> list[str]:
    nav = json.loads((DOCS / "docs.json").read_text(encoding="utf-8"))["navigation"]
    return [p for tab in nav["tabs"] for group in tab["groups"] for p in group["pages"]]


def sections(text: str, level: str) -> dict[str, str]:
    """Each `level` heading's text (backticks stripped) -> the section below it."""
    parts = re.split(rf"^{level} (.+)$", text, flags=re.MULTILINE)
    return {title.strip("` "): body for title, body in zip(parts[1::2], parts[2::2])}


# --- the site ----------------------------------------------------------------------------


def test_every_page_is_in_the_navigation_and_exists():
    pages = nav_pages()
    assert len(pages) == len(set(pages))
    on_disk = {p.relative_to(DOCS).with_suffix("").as_posix() for p in DOCS.rglob("*.mdx")}
    assert set(pages) == on_disk


@pytest.mark.parametrize("name", nav_pages())
def test_pages_have_a_title_and_description(name):
    front = re.match(r"---\n(.*?)\n---\n", page(name), re.DOTALL)
    assert front and re.search(r"^title: \S", front[1], re.M) and re.search(r"^description: \S", front[1], re.M)


@pytest.mark.parametrize("name", nav_pages())
def test_internal_links_resolve(name):
    pages = set(nav_pages())
    for target in re.findall(r"\]\((/[^)\s]*)\)|href=\"(/[^\"]*)\"", page(name)):
        path = (target[0] or target[1]).split("#")[0].strip("/") or "index"
        assert path in pages, f"{name} links to {path}"


@pytest.mark.parametrize("name", nav_pages())
def test_text_is_valid_mdx(name):
    """In MDX, a bare < starts a tag and a { an expression. Outside code and the components
    used, they must not appear."""
    text = re.sub(r"^---\n.*?\n---\n", "", page(name), flags=re.DOTALL)
    text = re.sub(r"^\s*```.*?^\s*```", "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"`[^`\n]+`", "", text)
    text = re.sub(r"</?[A-Z][A-Za-z]*(\s[^<>]*)?/?>", "", text)
    for char in "<>{}":
        line = next((ln for ln in text.splitlines() if char in ln), None)
        assert line is None, f"{name}: bare {char!r} in: {line}"


# --- pages that quote the code ----------------------------------------------------------


def test_the_consolidation_endpoint_page_matches_the_code():
    from openmemore import consolidate, llm
    from openmemore.models import CONVERSATION_BATCH_CHARS

    text = page("reference/consolidation-endpoint")
    assert consolidate.INSTRUCTIONS in text
    assert f'"max_output_tokens": {llm.MAX_OUTPUT_TOKENS}' in text and f'"max_tokens": {llm.MAX_OUTPUT_TOKENS}' in text
    assert f"{int(llm.TIMEOUT_S)} seconds" in text
    assert f"up to {consolidate.RELATED_MEMORIES} of your memories" in text
    assert f"{CONVERSATION_BATCH_CHARS:,} characters" in text
    assert consolidate.KEY_ENV in text and consolidate.KEY_FILE in text


def test_the_published_json_schema_matches_the_code():
    """Regenerate with: uv run python -c "import json; from openmemore.ingest.openmemore_format import json_schema;
    print(json.dumps(json_schema(), indent=2))" > schema/openmemore-export.schema.json"""
    from openmemore.ingest.openmemore_format import json_schema

    assert json.loads((DOCS.parent / "schema" / "openmemore-export.schema.json").read_text(encoding="utf-8")) == json_schema()


# --- reference coverage -----------------------------------------------------------------


def cli_commands():
    """(path, command) for every runnable command, groups excluded."""
    from typer._click.core import Context

    def walk(cmd, path):
        if hasattr(cmd, "list_commands"):
            ctx = Context(cmd, info_name=path[-1])
            for name in cmd.list_commands(ctx):
                yield from walk(cmd.get_command(ctx, name), [*path, name])
        elif not cmd.hidden:
            yield " ".join(path), cmd

    return list(walk(typer.main.get_command(app), ["openmemore"]))


@pytest.mark.parametrize(("path", "command"), cli_commands(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_cli_command_and_option_is_documented(path, command):
    section = sections(page("reference/cli"), "###").get(path)
    assert section is not None, f"no section for `{path}` in reference/cli.mdx"
    for param in command.params:
        if param.param_type_name == "option":
            if not param.hidden:
                assert f"`{param.opts[0]}" in section, f"{path}: {param.opts[0]} isn't documented"
        else:
            assert f"`{param.name.upper()}`" in section, f"{path}: {param.name.upper()} isn't documented"


def example_commands() -> list[tuple[str, str]]:
    """(page, line) for every openmemore command in the pages' shell examples, except
    templates with placeholders (<client>, MEMORY_ID, [--yes])."""
    found = []
    for name in nav_pages():
        for block in re.findall(r"```(?:sh|bash)\b[^\n]*\n(.*?)```", page(name), re.DOTALL):
            for line in block.splitlines():
                line = re.sub(r"\s+#.*$", "", line.strip())
                if line.startswith("openmemore ") and not re.search(r"[<\[]|\b[A-Z][A-Z_]{2,}\b", line):
                    found.append((name, line))
    return found


@pytest.mark.parametrize(("name", "line"), example_commands())
def test_example_commands_parse(name, line):
    """Each example names a real command, with options it takes, each given a valid value."""
    import shlex

    cmd = typer.main.get_command(app)
    ctx = cmd.make_context("openmemore", shlex.split(line)[1:])
    while hasattr(cmd, "resolve_command"):
        rest = [*getattr(ctx, "_protected_args", getattr(ctx, "protected_args", [])), *ctx.args]
        sub, cmd, rest = cmd.resolve_command(ctx, rest)
        ctx = cmd.make_context(sub, rest, parent=ctx)


def config_sections() -> list[tuple[str, str]]:
    """(section heading, key) for every config.toml key."""
    keys = []
    for name, field in Config.model_fields.items():
        kind = field.annotation
        if isinstance(kind, type) and issubclass(kind, BaseModel):
            keys += [(f"[{name}]", key) for key in kind.model_fields]
        elif get_origin(kind) is dict:
            keys += [(f"[{name}.<name>]", key) for key in get_args(kind)[1].model_fields]
        else:
            keys.append(("Top level", name))
    return keys


@pytest.mark.parametrize(("heading", "key"), config_sections())
def test_every_config_key_is_documented(heading, key):
    section = sections(page("reference/config"), "##").get(heading)
    assert section is not None, f"no section {heading} in reference/config.mdx"
    assert f"| `{key}` |" in section, f"{heading} {key} isn't documented"


@pytest.mark.anyio
async def test_every_mcp_tool_prompt_and_resource_is_documented(home):
    initialize(home)
    entries = sections(page("reference/mcp"), "###")
    async with connect(home) as client:
        for tool in (await client.list_tools()).tools:
            assert tool.name in entries, f"tool {tool.name} isn't documented"
            for argument in tool.input_schema.get("properties", {}):
                assert f'body="{argument}"' in entries[tool.name], f"{tool.name}({argument}) isn't documented"
            for result in (tool.output_schema or {}).get("properties", {}):
                if result != "notice":  # the stored-data notice, described once at the top
                    assert f'name="{result}"' in entries[tool.name], f"{tool.name} → {result} isn't documented"
        for prompt in (await client.list_prompts()).prompts:
            assert prompt.name in entries, f"prompt {prompt.name} isn't documented"
            for argument in prompt.arguments or []:
                assert f'body="{argument.name}"' in entries[prompt.name]
        for resource in (await client.list_resources()).resources:
            assert str(resource.uri) in entries, f"resource {resource.uri} isn't documented"
