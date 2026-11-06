import tempfile
import unittest
from pathlib import Path

from energyx.access import SnapshotDB
from energyx.agents.analysis_spec import AnalysisSpec
from energyx.agents.query_compiler import compile_analysis
from energyx.agents.runtime import EnergyXAgent
from energyx.tools import ArtifactStore
from energyx.tools.quality import profile_data_quality
from energyx.config import PATHS
from energyx.tools.registry import ToolContext

#: Several of these read the built snapshot.
needs_snapshot = unittest.skipUnless(
    PATHS.snapshot.exists(),
    "no snapshot built -- see SETUP.md section 7")

#: The structured planning engine is unfinished; `legacy` is the default.
structured_engine_incomplete = unittest.expectedFailure


class QueryCompilerTests(unittest.TestCase):
    def test_grouped_values_are_bound_and_identifiers_are_allowlisted(self):
        spec = AnalysisSpec(
            task="compare", entities={"country": "UK"},
            period={"start": "2017-01-01", "end": "2018-01-01"},
            measure="electricity_consumption", aggregation="mean",
            group_by=["city"], output={"kind": "table"}, resolution="hybrid",
        )

        query = compile_analysis(spec)

        self.assertNotIn("UK", query.sql)
        self.assertEqual(query.params[-3:], ["UK", "2017-01-01", "2018-01-01"])
        self.assertIn("GROUP BY city", query.sql)

    def test_unknown_measure_is_rejected_before_execution(self):
        spec = AnalysisSpec(task="aggregate", measure="value); DROP TABLE buildings")

        with self.assertRaisesRegex(ValueError, "measure"):
            compile_analysis(spec)

    @needs_snapshot
    def test_hourly_quantile_matches_an_independent_reference(self):
        spec = AnalysisSpec(
            task="aggregate", entities={"building_ids": ["Bear_education_Delia"]},
            period={"start": "2016-01-01", "end": "2017-01-01"},
            measure="electricity_consumption", aggregation="quantile",
            grain="hourly", parameters={"quantile": 0.9},
            output={"kind": "scalar"}, resolution="compute",
        )
        query = compile_analysis(spec)
        db = SnapshotDB()

        actual = db.value(query.sql, query.params)
        expected = db.value(
            "SELECT QUANTILE_CONT(value, 0.9) FROM electricity "
            "WHERE building_id = ? AND timestamp_local >= ? AND timestamp_local < ?",
            ["Bear_education_Delia", "2016-01-01", "2017-01-01"],
        )

        self.assertNotIn("Bear_education_Delia", query.sql)
        self.assertAlmostEqual(actual, expected, places=9)

    @needs_snapshot
    def test_structured_runtime_uses_compiler_without_sql_in_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = EnergyXAgent(
                stateless=True, planning_engine="structured",
                runs_root=Path(directory),
            ).ask("Compare average electricity consumption by city during 2017.")

        self.assertEqual(answer.plan["steps"][-1]["tool"], "run_structured_query")
        self.assertNotIn("sql", answer.plan["steps"][-1]["args"])
        self.assertEqual(answer.payload["kind"], "table")

    @needs_snapshot
    def test_quality_profile_applies_period_missingness_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore.for_run(Path(directory))
            result = profile_data_quality(
                ToolContext(SnapshotDB(), store), start="2017-01-01",
                end="2018-01-01", max_missing_fraction=0.02,
            )
            frame = store.get(result.handle)

        self.assertGreater(len(frame), 0)
        self.assertTrue((frame["missing_fraction"] < 0.02).all())
        self.assertEqual(result.evidence["period"], ["2017-01-01", "2018-01-01"])

    @needs_snapshot
    @structured_engine_incomplete
    def test_structured_runtime_routes_period_quality_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = EnergyXAgent(
                stateless=True, planning_engine="structured",
                runs_root=Path(directory),
            ).ask("Which buildings have less than 2% missing electricity data in 2017?")

        step = answer.plan["steps"][-1]
        self.assertEqual(step["tool"], "profile_data_quality")
        self.assertEqual(step["args"]["max_missing_fraction"], 0.02)
        self.assertEqual(answer.payload["evidence"]["period"],
                         ["2017-01-01", "2018-01-01"])

    @needs_snapshot
    @structured_engine_incomplete
    def test_structured_hourly_percentile_returns_scalar_value(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = EnergyXAgent(
                stateless=True, planning_engine="structured",
                runs_root=Path(directory),
            ).ask("What is the 90th percentile hourly demand for Bear_education_Delia in 2016?")

        self.assertEqual(answer.plan["steps"][-1]["tool"], "run_structured_query")
        self.assertEqual(answer.plan["answer_field"], "value")
        self.assertIsInstance(answer.payload["value"], float)


if __name__ == "__main__":
    unittest.main()
