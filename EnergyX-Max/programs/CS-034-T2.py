"""EnergyX-Max program for template CS-034-T2.

Compare the total count of {building_type} buildings to {building_type_2} buildings across all sites in {country}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    building_type_2 = slots['building_type_2']
    country = slots['country']

    # Get list of building IDs for the first building type in the specified country
    try:
        bldg_ids_type1 = data.buildings(building_type=building_type, country=country)
    except Exception:
        raise ValueError(f"No buildings found with type '{building_type}' in country '{country}'.")

    # Get list of building IDs for the second building type in the specified country
    try:
        bldg_ids_type2 = data.buildings(building_type=building_type_2, country=country)
    except Exception:
        raise ValueError(f"No buildings found with type '{building_type_2}' in country '{country}'.")

    # Count the buildings for each type
    count_type1 = len(bldg_ids_type1)
    count_type2 = len(bldg_ids_type2)

    return {
        "answer": {
            f"{building_type}": count_type1,
            f"{building_type_2}": count_type2
        },
        "unit": None
    }
