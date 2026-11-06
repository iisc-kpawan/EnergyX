"""Build the EnergyX knowledge graph from the catalog.

The graph is a projection, so this is idempotent and destructive by design: it
rebuilds from the catalog rather than merging, which is what makes a rebuild the
reset for anything a run asserted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyx.access import SnapshotDB                  # noqa: E402
from energyx.config import SNAPSHOT_VERSION            # noqa: E402
from energyx.knowledge.store import KnowledgeStore     # noqa: E402
from energyx.knowledge.vectors import VectorIndex      # noqa: E402


def main() -> int:
    db = SnapshotDB()
    store = KnowledgeStore(snapshot_version=SNAPSHOT_VERSION)
    report = store.build(db)
    print(f"\n  {report['triples']:,} triples -> {report['path']}")
    for section, count in report["sections"].items():
        print(f"    {section:<16} {count:>8,}")
    index = VectorIndex(snapshot_version=SNAPSHOT_VERSION)
    vectors = index.build(db)
    registered = store.add_facts(index.graph_triples())
    print(f"\n  {vectors['vectors']:,} vectors x {vectors['dimension']}d "
          f"-> {vectors['path']}")
    print(f"    registered as {registered:,} Representation triples")
    report["vectors"] = vectors

    print("\n  by class:")
    for klass, count in store.stats()["by_class"].items():
        print(f"    {klass:<20} {count:>6,}")
    (Path(report["path"]) / "BUILD.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
