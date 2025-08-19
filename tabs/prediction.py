import streamlit as st
import folium
from streamlit_folium import folium_static
import json
import tempfile
import os
import sys
import traceback
import io
import zipfile

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "aaa_lib_dgalPy"))

try:
    import dgalPy as dgal
    import sensorAssignmentModel as sa
    import muscat_wrappers as wr
    print("Successfully imported dgalPy and dependencies.")
    DGALPY_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    DGALPY_AVAILABLE = False
except Exception as e:
    print(f"General import error: {e}")
    traceback.print_exc()
    DGALPY_AVAILABLE = False


# -----------------------
# Session & helpers
# -----------------------
def initialize_prediction_session_state():
    if 'prediction_sensor_assignments' not in st.session_state:
        st.session_state.prediction_sensor_assignments = {}
    if 'custom_instance_id' not in st.session_state:
        st.session_state.custom_instance_id = "id-01"
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'model_completed' not in st.session_state:
        st.session_state.model_completed = False


def create_exact_sensor_name(sensor_spec: dict) -> str:
    sensor_model = (sensor_spec.get("model") or "").strip()
    sensor_manufacturer = (sensor_spec.get("manufacturer") or "").strip()
    exact_sensor_name = f"{sensor_model}_{sensor_manufacturer}"
    exact_sensor_name = exact_sensor_name.replace(" ", "_").replace("-", "_").replace(".", "")
    if len(exact_sensor_name) > 50:
        exact_sensor_name = sensor_model.replace(" ", "_").replace("-", "_").replace(".", "")
    if exact_sensor_name and exact_sensor_name[0].isdigit():
        exact_sensor_name = f"sensor_{exact_sensor_name}"
    return exact_sensor_name


def generate_minimal_pred_scenario(instance_id: str) -> dict:
    """
    Build ONLY the fields you want:
      sensors, title, config_file, detec_file_pettern, id
    """
    title_id = instance_id or "opt01"

    if not st.session_state.get('sensor_specifications'):
        return {
            "sensors": {},
            "title": title_id,
            "config_file": "muscat_input_var_core.json",
            "detec_file_pettern": "{sensor_name}_loc{location_id}_{uav_name}_detection.json",
            "id": title_id,
        }

    total_locations = len(st.session_state.potential_locations)
    location_to_index = {
        loc.get('name', f"Sensor {i+1}"): i
        for i, loc in enumerate(st.session_state.potential_locations)
    }

    # Zero vectors for all sensors
    sensor_placements = {}
    for sensor_spec in st.session_state.sensor_specifications:
        exact_name = create_exact_sensor_name(sensor_spec)
        sensor_placements[exact_name] = [0] * total_locations

    # Apply assignments
    if st.session_state.get('prediction_sensor_assignments'):
        for location_name, assignments in st.session_state.prediction_sensor_assignments.items():
            if location_name not in location_to_index:
                continue
            loc_idx = location_to_index[location_name]
            for assignment in assignments:
                spec = assignment['spec']
                exact_name = create_exact_sensor_name(spec)
                sensor_placements[exact_name][loc_idx] = 1

    return {
        "sensors": {k: {"actual_locs": v} for k, v in sensor_placements.items()},
        "title": title_id,
        "config_file": "muscat_input_var_core.json",
        "detec_file_pettern": "{sensor_name}_loc{location_id}_{uav_name}_detection.json",  # spelling per request
        "id": title_id,
    }


def run_model_with_memory_data(muscat_instance_json, memory_data):
    """Run model using detection ZIP from session state."""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_data = memory_data["zip_data"]
            zip_buffer = io.BytesIO(zip_data)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            var_core_path = os.path.join(temp_dir, "muscat_input_var_core.json")
            if not os.path.exists(var_core_path):
                return {"error": "muscat_input_var_core.json not found in detection data"}

            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                with open(var_core_path, 'r') as f:
                    muscatInputVarCore = json.load(f)

                wr.verifyInputVar(muscatInputVarCore)
                wr.verifyInputInstance(muscat_instance_json, muscatInputVarCore)

                inputVarCore = wr.wrapInputVarCore(muscatInputVarCore)
                inputInstance = wr.wrapInputInstance(muscat_instance_json, inputVarCore)

                output = sa.sensorMetrics(inputVarCore, inputInstance)
                return output
            finally:
                os.chdir(old_cwd)
    except Exception as e:
        return {"error": f"Error running model with memory data: {str(e)}"}


def run_model_with_auto_detection(muscat_instance_json):
    """Try session ZIP first, then common file paths on disk for var_core + detection files."""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}

    # Session ZIP
    if 'zip_data' in st.session_state and 'main_json' in st.session_state:
        memory_data = {
            "zip_data": st.session_state.zip_data,
            "main_json": st.session_state.main_json,
            "probability_files": st.session_state.get('probability_files', {})
        }
        return run_model_with_memory_data(muscat_instance_json, memory_data)

    # Fallback to disk
    try:
        config_file = muscat_instance_json.get("config_file", "muscat_input_var_core.json")
        possible_paths = [
            config_file,
            os.path.join("detection_outputs", config_file),
            os.path.join(".", config_file),
        ]
        var_core_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not var_core_path:
            return {"error": f"Configuration file '{config_file}' not found. Please run Detection Probability Analysis first."}

        old_cwd = os.getcwd()
        work_dir = os.path.dirname(var_core_path) or "."
        os.chdir(work_dir)
        try:
            config_filename = os.path.basename(var_core_path)
            with open(config_filename, 'r') as f:
                muscatInputVarCore = json.load(f)

            wr.verifyInputVar(muscatInputVarCore)
            wr.verifyInputInstance(muscat_instance_json, muscatInputVarCore)

            inputVarCore = wr.wrapInputVarCore(muscatInputVarCore)
            inputInstance = wr.wrapInputInstance(muscat_instance_json, inputVarCore)

            output = sa.sensorMetrics(inputVarCore, inputInstance)
            return output
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return {"error": f"Error running model with auto-detection: {str(e)}"}


# -----------------------
# Main tab
# -----------------------
def prediction_tab():
    """Assign sensors, then one-click ZIP download: pred_scenario + out_pred_scen."""
    initialize_prediction_session_state()
    st.header("Prediction - Custom Sensor Assignment")

    # Gate checks
    if not st.session_state.location_selected:
        st.warning("Please select a location in the initial screen first.")
        if st.button("Return to Location Selection"):
            st.session_state.location_selected = False
            st.rerun()
        return

    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return

    if not st.session_state.potential_locations:
        st.warning("Please place sensors in the Possible Sensor Placement tab first.")
        return

    if not st.session_state.sensor_specifications:
        st.warning("Please add sensor specifications in the sidebar configuration first.")
        return

    # Layout
    map_col, assignment_col = st.columns([3, 2])

    # Map
    with map_col:
        st.subheader("Sensor Locations Map")
        m = folium.Map(location=st.session_state.map_center, zoom_start=14)

        # Area boundary
        if st.session_state.boundary_type == "rectangle":
            folium.Rectangle(
                bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]],
                        [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                color='red', weight=2, fill=True, fill_color='red', fill_opacity=0.1
            ).add_to(m)
        elif st.session_state.boundary_type == "polygon":
            folium.Polygon(
                locations=st.session_state.boundary_points,
                color='red', weight=2, fill=True, fill_color='red', fill_opacity=0.1
            ).add_to(m)

        # Protected areas overlay (visual only if present)
        for i, area in enumerate(st.session_state.get('protected_areas', [])):
            folium.Polygon(
                locations=area['points'],
                color='green', weight=2, fill=True, fill_color='green', fill_opacity=0.2,
                tooltip=area.get('name', f"Protected Area {i+1}")
            ).add_to(m)

        # Sensor markers
        for i, location in enumerate(st.session_state.potential_locations):
            sensor_name = location.get('name', f"Sensor {i+1}")
            has_assignment = sensor_name in st.session_state.prediction_sensor_assignments
            assigned_sensors = st.session_state.prediction_sensor_assignments.get(sensor_name, [])
            marker_color = 'green' if (has_assignment and assigned_sensors) else 'red'
            popup_text = (
                f"{sensor_name}: {len(assigned_sensors)} sensor(s) assigned"
                if has_assignment and assigned_sensors else
                f"{sensor_name}: No sensors assigned"
            )
            folium.Marker(
                location=[location['lat'], location['lng']],
                popup=popup_text,
                tooltip=sensor_name,
                icon=folium.Icon(color=marker_color)
            ).add_to(m)

        folium_static(m, width=700, height=500)

    # Assignment + Download
    with assignment_col:
        st.subheader("Sensor Assignment")

        # Scenario ID used in filenames and inside JSON
        custom_id = st.text_input(
            "Enter Scenario ID:",
            value=st.session_state.custom_instance_id,
            key="custom_id_input"
        )
        st.session_state.custom_instance_id = custom_id
        instance_id = st.session_state.get('custom_instance_id', 'opt01')

        # Location & sensor selection UI
        location_names = [loc.get('name', f"Sensor {i+1}") for i, loc in enumerate(st.session_state.potential_locations)]
        selected_location = st.selectbox("Select Location:", location_names)

        sensor_options = [f"{spec['type']} - {spec['model']}" for spec in st.session_state.sensor_specifications]
        current_assignments = st.session_state.prediction_sensor_assignments.get(selected_location, [])

        if current_assignments:
            st.write("**Currently Assigned:**")
            for idx, assignment in enumerate(current_assignments):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"• {assignment['type']} - {assignment['model']}")
                with col2:
                    if st.button("Remove", key=f"remove_{selected_location}_{idx}"):
                        st.session_state.prediction_sensor_assignments[selected_location].pop(idx)
                        if not st.session_state.prediction_sensor_assignments[selected_location]:
                            del st.session_state.prediction_sensor_assignments[selected_location]
                        st.rerun()

        st.write("**Add New Sensor:**")
        selected_sensor_option = st.selectbox("Choose Sensor Type:", sensor_options, key="sensor_assignment_select")

        if st.button("Assign Sensor to Location"):
            sensor_type, sensor_model = selected_sensor_option.split(" - ", 1)
            sensor_spec = next((spec for spec in st.session_state.sensor_specifications
                                if spec['type'] == sensor_type and spec['model'] == sensor_model), None)
            if sensor_spec:
                if selected_location not in st.session_state.prediction_sensor_assignments:
                    st.session_state.prediction_sensor_assignments[selected_location] = []
                already_assigned = any(
                    a['type'] == sensor_type and a['model'] == sensor_model
                    for a in st.session_state.prediction_sensor_assignments[selected_location]
                )
                if not already_assigned:
                    st.session_state.prediction_sensor_assignments[selected_location].append({
                        'type': sensor_type,
                        'model': sensor_model,
                        'spec': sensor_spec
                    })
                    st.success(f"Assigned {sensor_type} - {sensor_model} to {selected_location}")
                    st.rerun()
                else:
                    st.warning(f"{sensor_type} - {sensor_model} is already assigned to {selected_location}")

        st.markdown("---")

        # One-click ZIP download (only appears if we have any assignments)
        if st.session_state.prediction_sensor_assignments:
            # Build minimal scenario JSON
            muscat_instance = generate_minimal_pred_scenario(instance_id)

            # Run model now so ZIP contains both files
            results = run_model_with_auto_detection(muscat_instance)

            if isinstance(results, dict) and "error" in results:
                st.error(f"Model run failed: {results['error']}")
            else:
                # Filter to GUARANTEE only the 5 allowed keys make it into pred_scenario.json
                allowed_keys = ["sensors", "title", "config_file", "detec_file_pettern", "id"]
                pred_scenario_filtered = {k: muscat_instance[k] for k in allowed_keys if k in muscat_instance}

                pred_scenario_str = json.dumps(pred_scenario_filtered, indent=4)
                out_pred_str = json.dumps(results, indent=4)

                pred_fname = f"pred_scenario_{instance_id}.json"
                out_fname  = f"out_pred_scen_{instance_id}.json"
                zip_fname  = f"prediction_scenario_{instance_id}.zip"

                # Create ZIP in memory
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(pred_fname, pred_scenario_str)
                    zf.writestr(out_fname, out_pred_str)

                st.download_button(
                    label="Download Scenario",
                    data=zip_buf.getvalue(),
                    file_name=zip_fname,
                    mime="application/zip",
                    key=f"dl_zip_only_{instance_id}"
                )

    # Upload-run section removed on purpose.
