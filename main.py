
import streamlit as st
from datetime import datetime
import json
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from tabs.map_selection import map_selection_tab
from tabs.sensor import sensor_tab
from tabs.protected_areas import protected_areas_tab
from tabs.detection_probability import detection_probability_tab
from tabs.prediction import prediction_tab
from tabs.display import display_tab
from tabs.fake_display import fake_display

st.set_page_config(page_title="UAV Sensor Detection Probability Calculator", layout="wide")

def geocode_location(location_name):
    try:
        geolocator = Nominatim(user_agent="uav_sensor_detection_app")
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        st.error(f"Geocoding error: {str(e)}")
        return None

def load_sensor_data():
    try:
        with open("sensor-data.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading sensor data: {str(e)}")
        return {"sensors": [], "uav_specifications": []}

def main():
    for key, default in {
        "area_selected": False,
        "sensors": [],
        "protected_areas": [],
        "grid_size": 0.01,
        "boundary_type": "rectangle",
        "boundary_points": [],
        "last_drawn_feature": None,
        "placement_mode": False,
        "current_sensor_location": None,
        "potential_locations": [],
        "sensor_specifications": [],
        "uav_specifications_list": [],
        "location_selected": False,
        "map_center": None,
        "detection_locked": False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if not st.session_state.location_selected:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>Search for a Location</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Enter a location (city, address, landmark, etc.)</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                location_input = st.text_input("Location", "", label_visibility="collapsed")
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Search Location") and location_input:
                    coordinates = geocode_location(location_input)
                    if coordinates:
                        st.session_state.map_center = coordinates
                        st.session_state.location_selected = True
                        st.success(f"Location found: {location_input} at coordinates {coordinates}")
                        st.rerun()
                    else:
                        st.error(f"Could not find coordinates for '{location_input}'.")
        return

    with st.sidebar:
        st.header("Configuration Parameters")

        if not st.session_state.detection_locked:
            sensor_selection_expanded = st.checkbox("Sensor Selection", value=False)
            if sensor_selection_expanded:
                sensor_data = load_sensor_data()
                if sensor_data["sensors"]:
                    sensor_types = [sensor["sensor_type"] for sensor in sensor_data["sensors"]]
                    sensor_tabs = st.tabs(sensor_types)
                    for i, tab in enumerate(sensor_tabs):
                        with tab:
                            st.write(f"### {sensor_types[i]} Sensors")
                            for sensor in sensor_data["sensors"][i]["parameters"]:
                                with st.expander(f"{sensor['model']} - {sensor['manufacturer']}"):
                                    st.write(f"**Description:** {sensor['description']}")
                                    st.write(f"**Detection Range:** {sensor['detection_range']} km")
                                    st.write(f"**Response Time:** {sensor['response_time']} seconds")
                                    st.write(f"**Price:** ${sensor['price_per_unit']:,}")
                                    if st.button("Add Sensor", key=f"add_{sensor['model']}"):
                                        st.session_state.sensor_specifications.append({
                                            "type": sensor_types[i],
                                            "detection_range": sensor["detection_range"],
                                            "response_time": sensor["response_time"],
                                            "model": sensor["model"],
                                            "manufacturer": sensor["manufacturer"],
                                            "price_per_unit": sensor["price_per_unit"],
                                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        st.success(f"Added {sensor['model']} sensor!")
                                        st.rerun()

            uav_selection_expanded = st.checkbox("UAV Specifications", value=False)
            if uav_selection_expanded:
                sensor_data = load_sensor_data()
                if sensor_data.get("uav_specifications"):
                    uav_types = [uav["uav_type"] for uav in sensor_data["uav_specifications"]]
                    selected_uav_type = st.selectbox("UAV Type", uav_types)
                    selected_uav = next((u for u in sensor_data["uav_specifications"] if u["uav_type"] == selected_uav_type), None)
                    if selected_uav:
                        altitude = st.slider("UAV Altitude (km)", float(selected_uav["altitude_range"][0]),
                                             float(selected_uav["altitude_range"][1]),
                                             float(np.mean(selected_uav["altitude_range"])), step=0.1)
                        speed = st.slider("UAV Speed (km/h)", float(selected_uav["speed_range"][0]),
                                          float(selected_uav["speed_range"][1]),
                                          float(np.mean(selected_uav["speed_range"])), step=1.0)

                        def generate_name(uav_type, alt, spd):
                            name = uav_type.lower().replace(" ", "_")
                            return f"{name}_alt_{str(alt).replace('.', '_')}_speed_{int(spd)}"

                        auto_name = generate_name(selected_uav_type, altitude, speed)
                        st.text_input("UAV Name", value=auto_name, disabled=True)

                        if st.button("Add UAV Configuration"):
                            if auto_name in [u["id"] for u in st.session_state.uav_specifications_list]:
                                st.error("This UAV configuration already exists.")
                            else:
                                st.session_state.uav_specifications_list.append({
                                    "id": auto_name,
                                    "type": selected_uav_type,
                                    "altitude": altitude,
                                    "speed": speed,
                                    "altitude_range": selected_uav["altitude_range"],
                                    "speed_range": selected_uav["speed_range"],
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                                st.success(f"Added UAV {auto_name}")
                                st.rerun()

        # ✅ Always show selected sensors
        st.subheader("Selected Sensors")
        if st.session_state.sensor_specifications:
            for i, sensor in enumerate(st.session_state.sensor_specifications):
                st.markdown(f"**{i+1}. {sensor['type']}** - {sensor['model']}")
                st.markdown(f"Range: {sensor['detection_range']}km, Response: {sensor['response_time']}s")
                st.markdown(f"Price: ${sensor['price_per_unit']:.2f}")
                if not st.session_state.detection_locked:
                    if st.button("Remove", key=f"remove_sensor_{i}"):
                        st.session_state.sensor_specifications.pop(i)
                        st.success("Sensor removed.")
                        st.rerun()
                st.markdown("---")
        else:
            st.markdown("No sensors added yet.")

        st.subheader("Added UAV Configurations")
        if st.session_state.uav_specifications_list:
            for i, uav in enumerate(st.session_state.uav_specifications_list):
                st.markdown(f"**{uav['id']}**: {uav['type']}")
                st.markdown(f"Altitude: {uav['altitude']}km, Speed: {uav['speed']}km/h")
                if not st.session_state.detection_locked:
                    if st.button("Remove", key=f"remove_uav_{i}"):
                        st.session_state.uav_specifications_list.pop(i)
                        st.success("UAV removed.")
                        st.rerun()
                st.markdown("---")
        else:
            st.markdown("No UAVs added yet.")

        if st.session_state.detection_locked:
            if st.button("🔁 Reset Detection Setup"):
                for key in [
                    "detection_locked", "detection_prob_calculated", "main_json", "probability_files",
                    "zip_data", "detection_zip_path", "sensor_specifications", "potential_locations",
                    "protected_areas", "uav_specifications_list", "all_uav_results"
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Reset complete. You can now reconfigure.")
                st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["Configuration Scenario", "Prediction Scenario", "Optimization Scenario", "Display"])

    with tab1:
        if not st.session_state.detection_locked:
            map_selection_tab()
            protected_areas_tab()
            sensor_tab()
            detection_probability_tab()
        else:
            st.success("Detection has been run. Inputs are locked. See the Display tab for the results.")

    with tab2:
        prediction_tab()

    with tab3:
        st.header("Optimization")
        st.info("This feature is coming in the next implementation step.")

    with tab4:
        mode = st.radio(
            "Choose Display Mode:",
            ["Configuration", "Prediction"],
            horizontal=True
        )

        if mode == "Configuration":
            fake_display()
        else:
            st.header("Prediction")
            st.info("This step needs to be implemented.")
            # TODO: Implement prediction flow here.


if __name__ == "__main__":
    main()
