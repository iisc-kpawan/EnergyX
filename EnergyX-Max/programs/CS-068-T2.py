"""EnergyX-Max program for template CS-068-T2.

Which {top_n} buildings in {country} have the highest 95th-percentile demand during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    top_n = slots.get('top_n')
    country = slots.get('country')
    month = slots.get('month')

    # Get building IDs for the specified country
    building_ids = data.buildings(country=country)

    if not building_ids:
        raise ValueError(f"No buildings found in {country}")

    # Define the time range for the month
    start, end = data.month_range(month)

    # Read energy readings for all selected buildings
    readings = data.readings(building_ids, start=start, end=end)

    # Filter out buildings that have no valid readings in this month
    # A building is considered to have data if it has at least one non-NaN value
    valid_readings_mask = ~readings.isna().any(axis=1)
    valid_building_ids = [bid for bid, valid in zip(building_ids, valid_readings_mask) if valid]

    if not valid_building_ids:
        raise ValueError(f"No buildings with valid readings found in {country} for {month}")

    # Calculate the 95th percentile demand for each valid building
    percentiles = readings.loc[:, valid_building_ids].quantile(0.95)

    # Sort by the 95th percentile value in descending order and get the top_n
    sorted_percentiles = percentiles.sort_values(ascending=False)
    top_buildings = sorted_percentiles.head(top_n).index.tolist()

    return {"answer": top_buildings, "unit": None}
