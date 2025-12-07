import streamlit as st
import pandas as pd

def render_levels_editor(config):
    """
    Renders an editor for the 'mappings.levels' section.
    Returns the updated levels dictionary.
    """
    st.subheader("Levels Configuration")
    
    levels_dict = config.get("mappings", {}).get("levels", {})
    
    # Convert to list of dicts for data editor if possible, or just use columns
    # Using columns for MVP as requested "E3, E4... should already be added. Option to add new fields"
    
    # We'll use a session state approach to track edits if we want to be fancy, 
    # but let's try to use standard widgets first. 
    # Actually, `st.data_editor` is perfect for this.
    
    data = [{"Level": k, "Rank": v} for k, v in levels_dict.items()]
    df = pd.DataFrame(data)
    
    # Editable dataframe
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        # Deprecation fix: use_container_width=True -> width="stretch"
        width="stretch",
        key="levels_editor",
        column_config={
            "Level": st.column_config.TextColumn("Level Name", required=True),
            "Rank": st.column_config.NumberColumn("Rank Value", required=True, min_value=0, step=1)
        }
    )
    
    # Reconstruct dictionary
    new_levels = {}
    for index, row in edited_df.iterrows():
        if row["Level"]: # Ensure not empty
            new_levels[row["Level"]] = int(row["Rank"])
            
    return new_levels

def render_location_targets_editor(config):
    """
    Renders an editor for 'mappings.location_targets'.
    Returns updated location_targets dictionary.
    """
    st.subheader("Location Targets")
    
    loc_dict = config.get("mappings", {}).get("location_targets", {})
    
    data = [{"City": k, "Tier/Rank": v} for k, v in loc_dict.items()]
    df = pd.DataFrame(data)
    
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        # Deprecation fix: use_container_width=True -> width="stretch"
        width="stretch",
        key="locations_editor",
        column_config={
            "City": st.column_config.TextColumn("City, State", required=True),
            "Tier/Rank": st.column_config.NumberColumn("Tier (Rank)", required=True, min_value=1, step=1)
        }
    )
    
    new_locs = {}
    for index, row in edited_df.iterrows():
        if row["City"]:
            new_locs[row["City"]] = int(row["Tier/Rank"])
            
    return new_locs

def render_location_settings_editor(config):
    """
    Renders slider for location settings.
    """
    st.subheader("Location Settings")
    
    loc_settings = config.get("location_settings", {})
    current_dist = loc_settings.get("max_distance_km", 50)
    
    new_dist = st.slider(
        "Max Distance (km) for Proximity Matching",
        min_value=0,
        max_value=200,
        value=current_dist,
        step=5
    )
    
    return {"max_distance_km": new_dist}

def render_config_ui(config):
    """
    Main entry point to render the full config UI.
    Returns: A NEW config dictionary with updates applied.
    """
    # Create a deep copy or just a new dict structure to avoid mutating input in place if we want purity
    # But often modifying existing structure is easier. 
    # Let's construct a new one to be safe.
    import copy
    new_config = copy.deepcopy(config)
    
    # Ensure structure exists
    if "mappings" not in new_config:
        new_config["mappings"] = {}
        
    # Levels
    new_config["mappings"]["levels"] = render_levels_editor(new_config)
    
    # Locations
    new_config["mappings"]["location_targets"] = render_location_targets_editor(new_config)
    
    # Settings
    new_config["location_settings"] = render_location_settings_editor(new_config)
    
    return new_config
