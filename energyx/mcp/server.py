"""MCP server: the same tool registry, exposed to external clients.

A **thin wrapper**, deliberately. Tools are plain Python functions called
in-process by the agents; this module re-describes that one registry
over JSON-RPC so another team's agent, an IDE, or a reviewer can use the energy
tools without EnergyX's agent layer.

Nothing is reimplemented here. If a tool changes signature, this server changes
with it, because both read the same `ToolSpec`.

Run:  python -m energyx.mcp.server         (stdio, JSON-RPC 2.0)
"""
from __future__ import annotations

import json
import sys
from typing import Any

from ..access import SnapshotDB
from ..config import SNAPSHOT_VERSION
from ..tools import ArtifactStore, ToolContext, load_all, registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "energyx", "version": "0.1.0"}


class MCPServer:
    """Minimal MCP over stdio: initialize, tools/list, tools/call."""

    def __init__(self, read_only: bool = True, expose_onboarding: bool = False):
        load_all()
        self.read_only = read_only
        self.expose_onboarding = expose_onboarding
        self._ctx: ToolContext | None = None

    # -- lazily opened, so `tools/list` costs nothing -----------------------
    @property
    def ctx(self) -> ToolContext:
        if self._ctx is None:
            self._ctx = ToolContext(
                db=SnapshotDB(), store=ArtifactStore.for_run(),
                snapshot_version=SNAPSHOT_VERSION,
                # An external client gets a read-only view by default: it may
                # query the graph but not assert facts into it.
                options={"write_back": not self.read_only})
        return self._ctx

    def specs(self) -> list:
        specs = registry.runtime_tools()
        if self.expose_onboarding:
            specs = specs + registry.for_agent("onboarding")
        if self.read_only:
            specs = [s for s in specs if s.name != "assert_findings"]
        return sorted(specs, key=lambda s: s.name)

    # -- protocol ----------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        request_id = message.get("id")
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL_VERSION,
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": SERVER_INFO,
                          "instructions":
                              "EnergyX electricity analytics over a frozen "
                              f"snapshot ({SNAPSHOT_VERSION}). Resolve an "
                              "entity set with find_buildings or the knowledge "
                              "graph before fetching measurements."}
            elif method in ("notifications/initialized", "initialized"):
                return None                       # notification: no reply
            elif method == "tools/list":
                result = {"tools": [self._describe(s) for s in self.specs()]}
            elif method == "tools/call":
                result = self._call(message.get("params") or {})
            elif method == "ping":
                result = {}
            else:
                return _error(request_id, -32601, f"unknown method {method!r}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # noqa: BLE001 - a server must not die on input
            return _error(request_id, -32603, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _describe(spec) -> dict:
        return {"name": spec.name,
                "description": f"[{spec.agent}] {spec.summary}",
                "inputSchema": spec.parameters}

    def _call(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        allowed = {s.name for s in self.specs()}
        if name not in allowed:
            return _tool_error(f"unknown or unavailable tool {name!r}")
        problems = registry.validate_args(name, arguments)
        if problems:
            return _tool_error("; ".join(problems))
        result = registry.call(name, self.ctx, **arguments)
        payload = result.to_payload()
        return {"content": [{"type": "text",
                             "text": json.dumps(payload, default=str)[:60000]}],
                "isError": result.status.value == "ERROR",
                "structuredContent": payload}

    # -- transport ---------------------------------------------------------
    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                _write(stdout, _error(None, -32700, "parse error"))
                continue
            response = self.handle(message)
            if response is not None:
                _write(stdout, response)


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _tool_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _write(stream, payload: dict) -> None:
    stream.write(json.dumps(payload, default=str) + "\n")
    stream.flush()


if __name__ == "__main__":
    MCPServer().serve()
