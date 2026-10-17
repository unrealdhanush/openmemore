import tomllib

from typer.testing import CliRunner

from openmcp import config, db
from openmcp.cli import app, initialize

runner = CliRunner()


def test_init_creates_layout(home):
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (home / "openmcp.db").is_file()
    assert (home / "config.toml").is_file()
    assert (home / "imports").is_dir()
    assert (home / "logs").is_dir()


def test_default_config_contents(home):
    initialize(home)
    with (home / "config.toml").open("rb") as f:
        cfg = tomllib.load(f)
    assert cfg == config.DEFAULT_CONFIG


def test_init_is_idempotent_and_keeps_user_config(home):
    initialize(home)
    custom = '[defaults]\nread_scopes = ["mine"]\nwrite_scope = "mine"\n'
    (home / "config.toml").write_text(custom)

    second = initialize(home)

    assert second.migrations_applied == []
    assert second.config_written is False
    assert (home / "config.toml").read_text() == custom


def test_init_db_is_migrated_and_wal(home):
    initialize(home)
    conn = db.connect(config.db_path(home))
    assert db.schema_version(conn) == max(v for v, _ in db.migrations())
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_home_defaults_to_dot_openmcp(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENMCP_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.home() == tmp_path / ".openmcp"
