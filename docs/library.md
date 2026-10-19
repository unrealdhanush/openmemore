# Using OpenMCP from your own programs

> **Experimental.** The import name changes when the project gets its final name (SPEC.md M9).

Your own agents and scripts can share memory with your AI assistants. They read what Claude or Cursor saved, and write what your assistants will find later. This runs in your program's process, with no server:

```python
from openmcp import Memory

memory = Memory("trip-planner")

@memory.recall(query="request")       # before each call: related memories and chats, as `context`
@memory.remember(question="request")  # after each call: the request and the result, stored as an exchange
def plan(request: str, context: str = "") -> str:
    ...  # call your model with `context` in the prompt
```

It uses the same store and rules as your assistants: `~/.openmcp` (or `OPENMCP_HOME`), scopes, the secret filter, and the audit log. Every call is logged under the program's client name, with the app recorded as `openmcp library`.

## `Memory(name, *, home=None)`

- **`name`:** the client name, normalized like an app's name (`"Trip Planner"` becomes `trip-planner`). Give it scopes in `config.toml` like any client:

  ```toml
  [clients.trip-planner]
  read_scopes = ["personal"]
  write_scope = "personal"
  ```

  Without a section, it uses `[defaults]`.
- **`home`:** another data directory, for example in tests.

## Reading

| Call | Returns |
|---|---|
| `memory.search(query, *, limit=10, kinds=(), project=None, chats=True)` | a list of `Hit`: related memories and conversation messages, matched by meaning and by words |
| `memory.context(query, *, limit=5, kinds=(), profile=False)` | a block of text ready for a prompt: the stored-data notice, then each related item wrapped. Empty when nothing is related |
| `memory.profile(project=None)` | preferences, facts, active projects and recent decisions, the same as the `get_profile` tool |
| `memory.recent(limit=5, source=None)` | the latest conversations, the same as the `recent_conversations` tool |

A `Hit` has these fields: `type` (`"memory"` or `"chat"`), `id`, `text`, `wrapped`, `source`, `score`, `kind`, `subject`, `project`, `conversation_id`, `title` and `created_at`.
- **`text`** is the stored text, for your program's logic.
- **`wrapped`** is the same text labelled as stored data. Use it in a prompt: the label tells the model to treat it as information, not instructions.

`profile()` and `recent()` return their content wrapped, as the MCP tools do.

## Writing

| Call | Does |
|---|---|
| `memory.save(content, *, kind="note", subject=None, project=None)` | saves a memory in this client's write scope. Returns its id, or `None` if the same statement is already stored |
| `memory.record(question, answer, *, conversation=None, title=None)` | stores an exchange, with secrets replaced by `[REDACTED]`. Returns the conversation id |
| `memory.update(id, content)` | corrects a memory. The old version is kept as history. Returns the new id |
| `memory.forget(id)` | deletes a memory and its history. Returns how many versions were deleted |

`kind` is one of `fact`, `preference`, `project`, `decision` or `note`.

## Decorators

Both work on plain and `async` functions. For `async` ones, the store work runs in a thread.

- **`@memory.remember(question=None, conversation=None)`** stores each call as an exchange.
  - **Question:** the argument named `question`; otherwise the first string argument; otherwise all the arguments.
  - **Answer:** the return value, as text. Non-strings are stored as JSON.
  - **Exceptions:** if the function raises, nothing is stored and the exception passes through.
- **`@memory.recall(query=None, into="context", limit=5, kinds=(), profile=False)`** passes `memory.context(...)` into the argument named by `into` before each call.
  - **Query:** the argument named `query`, or otherwise the first string argument.
  - **Checked up front:** the function must accept `into`, which is checked when it's decorated.

Both can be used bare (`@memory.remember`) or with arguments (`@memory.remember(question="request")`).

### Conversations

Calls are grouped into conversations:
- **By default:** one conversation per decorated function, per `Memory`, per run of your program.
- **`conversation="trip-42"`** on `remember` or `record` names a group.
- **`with memory.conversation("trip-42"):`** groups everything recorded inside it, across functions.

## Errors

- **Direct calls** raise `openmcp.OpenMCPError` with the reason: not found (or not in this client's scopes), a secret, and so on.
- **Decorators** never break your function. If storing or recalling fails, the problem is logged on the `openmcp` logger, and the function runs as usual (with an empty `context` for `recall`).

## Cost

`import openmcp` loads nothing heavy. The store opens on the first call, and the embedding model loads only when something needs a vector. Without the model, from `openmcp embed`, search uses words only, and new items are embedded later by the daemon.
