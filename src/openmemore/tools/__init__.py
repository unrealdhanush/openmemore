"""Tool logic. Each function takes a connection, the calling client, and validated
arguments, and returns a result model. Expected failures raise ToolError, whose
message reaches the calling model."""

from __future__ import annotations

from dataclasses import dataclass

from mcp.server.mcpserver.exceptions import ToolError

from openmemore.config import ClientScopes
from openmemore.models import Memory


@dataclass(frozen=True)
class Caller:
    client: str
    scopes: ClientScopes

    def can_read(self, scope: str) -> bool:
        return self.scopes.read_scopes is None or scope in self.scopes.read_scopes

    def can_write(self, scope: str) -> bool:
        """New items go to write_scope; with ignore_scopes (read_scopes None), existing
        items in any scope can be changed too."""
        return self.scopes.read_scopes is None or scope == self.scopes.write_scope


def require_writable(caller: Caller, memory: Memory | None, memory_id: int) -> Memory:
    """The memory, if this client may both read and write it. Memories in scopes the
    client can't read are reported as not found, so their existence doesn't leak."""
    if memory is None or not caller.can_read(memory.scope):
        raise ToolError(f"memory {memory_id} not found")
    if not caller.can_write(memory.scope):
        raise ToolError(
            f"memory {memory_id} is in scope '{memory.scope}', which this client can't write to "
            f"(it writes to '{caller.scopes.write_scope}')"
        )
    return memory
