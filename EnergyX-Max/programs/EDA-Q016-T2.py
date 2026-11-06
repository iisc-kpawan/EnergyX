"""EnergyX-Max program for template EDA-Q016-T2.

What is the total number of buildings with at least {coverage_pct}% valid electricity readings in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, coverage_pct, year):
    # Determine the start and end timestamps for the given year
    start_ts, end_ts = data.year_range(year)
    
    # Get all building IDs from the metadata
    all_building_ids = data.buildings()
    
    if not all_building_ids:
        raise ValueError("No buildings found in the dataset.")

    # Filter buildings that have at least `coverage_pct` valid readings for the year
    qualifying_buildings = data.qualifying(
        building_ids=all_building_ids, 
        year=year, 
        min_share=coverage_pct / 100.0
    )
    
    # The function returns a list of building IDs that qualify
    # We need to return the count (total number) of these buildings
    count = len(qualifying_buildings)
    
    return {"answer": count, "unit": None}
