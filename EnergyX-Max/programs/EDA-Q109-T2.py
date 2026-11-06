"""EnergyX-Max program for template EDA-Q109-T2.

Which days during Q4 {year} show {building_id}'s daily consumption more than {sd_multiplier} standard deviations from its typical daily average, excluding holidays?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def answer(data, year, building_id, sd_multiplier):
    # Get readings for the specific building and Q4 of the given year
    start_date = f"{year}-10-01"
    end_date = f"{year + 1}-01-01"
    
    try:
        readings_df = data.readings([building_id], start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if readings_df.empty or building_id not in readings_df.columns:
        raise ValueError(f"No data found for building {building_id} in Q4 {year}.")

    # Extract the single column of readings for this building
    series = readings_df[building_id]
    
    # Ensure DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    # Define Q4 dates explicitly
    q4_start = datetime(year, 10, 1)
    q4_end = datetime(year + 1, 1, 1)
    
    # Filter the series to strictly Q4 dates
    mask = (series.index >= q4_start) & (series.index < q4_end)
    q4_series = series[mask]
    
    if len(q4_series) == 0:
        raise ValueError(f"No valid hours found for building {building_id} in Q4 {year}.")

    # Define US holidays for the year using standard library logic to avoid external dependency
    # US Federal Holidays
    us_holidays = set()
    
    # New Year's Day (Jan 1) - not in Q4
    # Martin Luther King Jr. Day (3rd Monday Jan) - not in Q4
    # Presidents' Day (3rd Mon Feb) - not in Q4
    # Memorial Day (last Mon May) - not in Q4
    # Juneteenth (Jun 19) - not in Q4
    # Independence Day (Jul 4) - not in Q4
    # Labor Day (1st Mon Sep) - not in Q4
    # Columbus Day (2nd Mon Oct) - IN Q4
    # Veterans Day (Nov 11) - IN Q4
    # Thanksgiving (4th Thu Nov) - IN Q4
    # Christmas (Dec 25) - IN Q4
    
    # Calculate specific dates for holidays in Q4
    # Columbus Day: 2nd Monday of October
    oct_1 = datetime(year, 10, 1).weekday()
    # First Monday of Oct is if oct_1 <= 0. If not, first Monday is (oct_1 + 7 - 1) or similar logic.
    # Simpler: iterate days in Q4 and check specific rules
    
    q4_dates = []
    current = datetime(year, 10, 1)
    while current < q4_end:
        q4_dates.append(current)
        current += timedelta(days=1)
    
    holiday_set = set()
    
    # Check for Columbus Day (2nd Monday Oct)
    oct_monday_1 = datetime(year, 10, 1).weekday()
    if oct_monday_1 <= 0:
        first_mon_oct = datetime(year, 10, 1 + (0 - oct_monday_1))
    else:
        first_mon_oct = datetime(year, 10, 1 + (7 - oct_monday_1))
    
    second_mon_oct = first_mon_oct + timedelta(days=7)
    if second_mon_oct.month == 10 and second_mon_oct.day <= 31:
        holiday_set.add(second_mon_oct.date())
        
    # Veterans Day (Nov 11)
    nov_11 = datetime(year, 11, 11).date()
    if nov_11 in [d.date() for d in q4_dates]:
        holiday_set.add(nov_11)
        
    # Thanksgiving (4th Thursday Nov)
    nov_thursday_1 = datetime(year, 11, 1).weekday()
    if nov_thursday_1 <= 3: # 0=Mon, ..., 3=Thu
        first_thu_nov = datetime(year, 11, 1 + (3 - nov_thursday_1))
    else:
        first_thu_nov = datetime(year, 11, 1 + (14 - nov_thursday_1)) # Next month logic not needed as we know Nov has 4 Thurs
        
    fourth_thu_nov = first_thu_nov + timedelta(days=21) # 3 weeks later
    if fourth_thu_nov.month == 11:
        holiday_set.add(fourth_thu_nov.date())
        
    # Christmas (Dec 25)
    dec_25 = datetime(year, 12, 25).date()
    if dec_25 in [d.date() for d in q4_dates]:
        holiday_set.add(dec_25)
        
    # Filter out holidays from the Q4 series index
    valid_mask = ~q4_series.index.isin(holiday_set)
    valid_dates = q4_series[valid_mask]
    
    if len(valid_dates) == 0:
        raise ValueError(f"All days in Q4 {year} for building {building_id} are holidays or missing.")

    # Aggregate to daily totals for non-holiday days
    daily_totals = valid_dates.resample('D').sum()
    
    if len(daily_totals) == 0:
        raise ValueError("No daily consumption data available after excluding holidays.")

    typical_avg = daily_totals.mean()
    std_dev = daily_totals.std()
    
    if pd.isna(std_dev):
        raise ValueError("Standard deviation cannot be calculated (constant or single value).")

    threshold = typical_avg + (sd_multiplier * std_dev)
    
    # Find days where consumption > threshold
    outlier_mask = daily_totals > threshold
    outlier_days = daily_totals[outlier_mask].index
    
    # Convert to date strings YYYY-MM-DD and sort
    result_dates = sorted([str(d.date()) for d in outlier_days])

    return {"answer": result_dates, "unit": None}
