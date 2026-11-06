"""EnergyX-Max program for template EDA-Q009-T2.

For buildings in {country}, what is the percentage of hours with valid electricity readings during 2016-2017?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    country = slots.get("country")
    
    if not country:
        raise ValueError("Country slot is required.")
    
    # Get building IDs for the specified country
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in {country}.")
        
    if not building_ids:
        raise ValueError(f"No buildings found in {country}.")

    # Define the time range for 2016-2017
    start_2016, end_2016 = data.year_range(2016)
    start_2017, end_2017 = data.year_range(2017)
    
    # Calculate the total hours in 2016 and 2017
    hours_2016 = (end_2016 - start_2016).total_seconds() / 3600
    hours_2017 = (end_2017 - start_2017).total_seconds() / 3600
    total_hours = hours_2016 + hours_2017
    
    if total_hours <= 0:
        raise ValueError("Invalid time range calculation.")

    # Readings for all buildings in the country for the specified period
    readings_df = data.readings(building_ids, start=start_2016, end=end_2017)
    
    if readings_df.empty:
        raise ValueError("No reading data available for the specified period.")

    # Calculate total valid hours across all buildings and the total possible hours
    # We sum the count of non-null values per building. 
    # Note: The DataFrame has one column per building.
    valid_counts = readings_df.notna().sum(axis=0)
    total_valid_hours = valid_counts.sum()
    
    if total_valid_hours == 0:
        raise ValueError("No valid electricity readings found for the specified period.")

    percentage = (total_valid_hours / total_hours) * 100
    
    return {"answer": round(percentage, 6), "unit": "%"}
