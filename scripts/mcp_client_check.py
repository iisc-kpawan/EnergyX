"""Verify the MCP server with an external client.

Speaks JSON-RPC to the server as a separate process, exactly as a third-party
client would -- the point being that the energy tools are usable without
EnergyX's agent layer.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rpc(process, message: dict) -> dict | None:
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    if "id" not in message:
        return None
    return json.loads(process.stdout.readline())


def main() -> int:
    process = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m", "energyx.mcp.server"],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        init = rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2024-11-05"}})
        print("  initialize ->", init["result"]["serverInfo"],
              "protocol", init["result"]["protocolVersion"])
        rpc(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        listed = rpc(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = listed["result"]["tools"]
        print(f"  tools/list -> {len(tools)} tools")
        print("    e.g.", tools[0]["name"], "|", tools[0]["description"][:64])
        assert all("inputSchema" in t for t in tools), "every tool needs a schema"

        called = rpc(process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                               "params": {"name": "search_catalog",
                                          "arguments": {"entity": "countries"}}})
        payload = called["result"]["structuredContent"]
        print("  tools/call search_catalog ->", payload["status"],
              [r["country"] for r in payload.get("data", [])])

        bad = rpc(process, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "rank_buildings",
                                       "arguments": {"building_ids": "not-a-list"}}})
        print("  bad arguments rejected ->", bad["result"]["isError"],
              "|", bad["result"]["content"][0]["text"][:70])

        blocked = rpc(process, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                "params": {"name": "assert_findings",
                                           "arguments": {"findings": [], "method": "x"}}})
        print("  write tool withheld from a read-only client ->",
              blocked["result"]["isError"])

        unknown = rpc(process, {"jsonrpc": "2.0", "id": 6, "method": "nope"})
        print("  unknown method ->", unknown["error"]["message"])

        # 54 runtime tools less assert_findings, which a read-only client
        # must not see.
        expected = 53
        ok = (len(tools) == expected and payload["status"] == "OK"
              and bad["result"]["isError"] and blocked["result"]["isError"]
              and "assert_findings" not in {t["name"] for t in tools})
        if len(tools) != expected:
            print(f"  expected {expected} tools in read-only mode, got {len(tools)}")
        print(f"\n  MCP server usable by an external client: {ok}")
        return 0 if ok else 1
    finally:
        process.stdin.close()
        process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
