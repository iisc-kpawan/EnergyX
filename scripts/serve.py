"""Run the EnergyX service.

Binds to 127.0.0.1 by default: this is a shared machine, and the snapshot is
unpublished work. Reach it from a laptop with an SSH tunnel:

    ssh -L 8321:localhost:8321 <user>@<host>
    open http://localhost:8321
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-supervisor", action="store_true",
                        help="run in this process; a crash is then fatal")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"  WARNING: binding {args.host} exposes an unauthenticated "
              f"service on a shared machine.", file=sys.stderr)

    print(f"  EnergyX -> http://{args.host}:{args.port}")
    if args.reload or args.no_supervisor:
        import uvicorn
        uvicorn.run("energyx.service.app:app", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")
        return 0
    return supervise(args)


def supervise(args) -> int:
    """Run the server as a child and restart it if it dies unexpectedly.

    A native crash in a C extension cannot be caught in Python: the no-raise
    contract covers exceptions, and a segfault is not one. One was observed
    inside a pyoxigraph SPARQL query, taking the whole service with it and
    every session on it. The query shape that triggered it is guarded,
    but the crash was never reproduced on demand, so the honest assumption is
    that another one exists.

    A supervised worker turns "the app is gone until someone notices" into a
    few seconds of downtime. Restarts are capped, so a server that cannot start
    at all fails loudly instead of looping.
    """
    import signal
    import subprocess
    import time

    command = [sys.executable, "-m", "uvicorn", "energyx.service.app:app",
               "--host", args.host, "--port", str(args.port),
               "--log-level", "info"]
    restarts, window_start = 0, time.time()
    child: subprocess.Popen | None = None

    def relay(signum, _frame):
        if child and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, relay)
    signal.signal(signal.SIGINT, relay)

    while True:
        started = time.time()
        child = subprocess.Popen(command)
        code = child.wait()
        ran_for = time.time() - started

        if code == 0 or code == -signal.SIGTERM or code == -signal.SIGINT:
            return 0                            # asked to stop

        if time.time() - window_start > 600:    # a fresh hour of credit
            restarts, window_start = 0, time.time()
        restarts += 1
        if restarts > 5:
            print(f"  worker died {restarts} times in 10 minutes "
                  f"(last exit {code}); not restarting again", file=sys.stderr)
            return 1
        signal_name = (f"signal {-code}" if code < 0 else f"exit {code}")
        print(f"  worker died after {ran_for:.0f}s ({signal_name}); "
              f"restarting [{restarts}/5]", file=sys.stderr)
        time.sleep(min(2 ** restarts, 15))


if __name__ == "__main__":
    raise SystemExit(main())
