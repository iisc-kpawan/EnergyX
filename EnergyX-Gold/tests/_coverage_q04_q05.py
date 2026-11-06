"""Raw-file coverage counts shared by the Q04 and Q05 checks. Pandas only."""

from __future__ import annotations

import pandas as pd


def coverage(raw, year: int) -> pd.DataFrame:
    """Per building with electricity, for one year: valid electricity hours, those hours that also
    have a valid air temperature at its site, and the site's valid air temperature hours."""
    elec = raw.electricity_cleaned
    elec = elec[elec.index.year == year].notna()
    weather = raw.weather[(raw.weather["timestamp"].dt.year == year) & raw.weather["airTemperature"].notna()]
    temp = (weather.assign(ok=True).pivot_table(index="timestamp", columns="site_id", values="ok", aggfunc="any")
            .reindex(elec.index).fillna(False).astype(bool))
    site_of = raw.metadata.set_index("building_id")["site_id"]
    rows = []
    for building in elec.columns:
        site = site_of[building]
        site_temp = temp[site] if site in temp else pd.Series(False, index=elec.index)
        rows.append({
            "building_id": building,
            "elec_hours": int(elec[building].sum()),
            "elec_hours_with_temp": int((elec[building] & site_temp).sum()),
            "site_temp_hours": int(site_temp.sum()),
        })
    return pd.DataFrame(rows).set_index("building_id")
