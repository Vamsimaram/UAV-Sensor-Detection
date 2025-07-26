import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import json
import tempfile
import os
import sys
import traceback
import re

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


def initialize_prediction_session_state():
    """Initialize all prediction-related session state variables"""
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'model_completed' not in st.session_state:
        st.session_state.model_completed = False
    if 'prediction_sensor_assignments' not in st.session_state:
        st.session_state.prediction_sensor_assignments = {}
    if 'custom_instance_id' not in st.session_state:
        st.session_state.custom_instance_id = "id-01"


def create_exact_sensor_name(sensor_spec):
    """
    Create exact sensor name using the same logic as detection_probability.py
    This ensures consistency between detection and prediction files
    """
    sensor_model = sensor_spec["model"].strip()
    sensor_manufacturer = sensor_spec.get("manufacturer", "").strip()
    
    # Create exact sensor name using model and manufacturer
    exact_sensor_name = f"{sensor_model}_{sensor_manufacturer}".replace(" ", "_").replace("-", "_").replace(".", "")
    
    # If the exact name is too long, use just the model name
    if len(exact_sensor_name) > 50:
        exact_sensor_name = sensor_model.replace(" ", "_").replace("-", "_").replace(".", "")
    
    # Ensure the name doesn't start with a number (JSON key requirement)
    if exact_sensor_name[0].isdigit():
        exact_sensor_name = f"sensor_{exact_sensor_name}"
    
    return exact_sensor_name


def check_for_saved_detection_files():
    """Check if detection files are available from previous session (in memory)"""
    # Check if we have detection data in session state (in-memory)
    if 'zip_data' in st.session_state and 'main_json' in st.session_state:
        return {"type": "memory", "data": {
            "zip_data": st.session_state.zip_data,
            "main_json": st.session_state.main_json,
            "probability_files": st.session_state.get('probability_files', {})
        }}
    
    # Check for individual files in session state
    if 'main_json' in st.session_state and 'probability_files' in st.session_state:
        return {"type": "session", "data": {
            "var_core": st.session_state.main_json,
            "detection_files": st.session_state.probability_files
        }}
    
    return None


def run_model_with_memory_data(muscat_instance_json, memory_data):
    """Run model using in-memory detection data from session state"""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}
    
    try:
        import zipfile
        import io
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP data from memory to temporary directory
            zip_data = memory_data["zip_data"]
            
            # Create a file-like object from the zip data
            zip_buffer = io.BytesIO(zip_data)
            
            # Extract contents
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the var_core file
            var_core_path = os.path.join(temp_dir, "muscat_input_var_core.json")
            
            if not os.path.exists(var_core_path):
                return {"error": "muscat_input_var_core.json not found in detection data"}
            
            # Change to temp directory for detection file access
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Load and run the model
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


def run_model_with_session_data(muscat_instance_json, session_data):
    """Run model using session state data"""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write var_core JSON to temp directory
            var_core_path = os.path.join(temp_dir, "muscat_input_var_core.json")
            with open(var_core_path, 'w') as f:
                f.write(session_data["var_core"])
            
            # Write detection files to temp directory
            for filename, content in session_data["detection_files"].items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
            
            # Change to temp directory for detection file access
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Load and run the model
                with open("muscat_input_var_core.json", 'r') as f:
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
        return {"error": f"Error running model with session data: {str(e)}"}


def run_model_with_auto_detection(muscat_instance_json):
    """Run model with automatic detection file loading"""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}
    
    # First, try to use in-memory detection data from current session
    if 'zip_data' in st.session_state and 'main_json' in st.session_state:
        memory_data = {
            "zip_data": st.session_state.zip_data,
            "main_json": st.session_state.main_json,
            "probability_files": st.session_state.get('probability_files', {})
        }
        return run_model_with_memory_data(muscat_instance_json, memory_data)
    
    # Fallback: search file system
    try:
        config_file = muscat_instance_json.get("config_file", "muscat_input_var_core.json")
        
        # Look for the config file in common locations
        possible_paths = [
            config_file,
            os.path.join("detection_outputs", config_file),
            os.path.join(".", config_file),
        ]
        
        var_core_path = None
        
        # Try to find the config file directly
        for path in possible_paths:
            if os.path.exists(path):
                var_core_path = path
                break
        
        if not var_core_path:
            return {"error": f"Configuration file '{config_file}' not found. Please run Detection Probability Analysis first."}
        
        # Change to the directory containing the config file
        old_cwd = os.getcwd()
        work_dir = os.path.dirname(var_core_path) or "."
        os.chdir(work_dir)
        
        try:
            # Load and run the model
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


def enhanced_prediction_upload_section():
    """Simplified upload section with auto-detection and persistent results"""
    st.subheader("Upload Files to Run Analytical Model")
    
    # Check for auto-available files in memory (session state)
    saved_files = check_for_saved_detection_files()
    
    if saved_files:
        st.success("Detection data found in current session!")
        
        # Simplified choice: just auto-detect vs manual
        use_saved = st.radio(
            "File Source:",
            ["Auto-detect from scenario", "Upload files manually"],
            key="file_source_choice",
            help="Auto-detect will use current session data first, then search for files if needed"
        )
        
        if use_saved == "Auto-detect from scenario":
            uploaded_opt_json = st.file_uploader(
                "Upload Prediction Scenario", 
                type=["json"],
                help="System will automatically use detection data from current session or search for files"
            )
            
            if uploaded_opt_json:
                # Preview the uploaded file to show the config reference
                try:
                    muscat_instance_json = json.load(uploaded_opt_json)
                    uploaded_opt_json.seek(0)  # Reset for later use
                    
                    config_file = muscat_instance_json.get("config_file")
                    if config_file:
                        st.success(f"Found config reference: `{config_file}`")
                    else:
                        st.info("No 'config_file' reference found. Will use 'muscat_input_var_core.json' by default.")
                    
                    if st.button("Run Model (Auto-Detection)", type="primary", key="run_model_auto"):
                        with st.spinner("Running model with session data and auto-detection..."):
                            results = run_model_with_auto_detection(muscat_instance_json)
                            
                            if results:
                                if "error" in results:
                                    st.error(f"Error: {results['error']}")
                                    st.info("Tip: Make sure you've run the Detection Probability Analysis first in this session.")
                                else:
                                    # IMPORTANT: Store results in session state
                                    st.session_state.model_results = results
                                    st.session_state.model_completed = True
                                    # st.success("Model completed successfully! Results generated.")
                                    
                                    # Show download button for JSON results
                                    results_json_string = json.dumps(results, indent=4)
                                    instance_id = st.session_state.get('custom_instance_id', 'opt01')
                                    output_filename = f"out_prediction_scenario_{instance_id}.json"
                                    
                                    # st.download_button(
                                    #     label="Download Prediction Results JSON",
                                    #     data=results_json_string,
                                    #     file_name=output_filename,
                                    #     mime="application/json",
                                    #     type="primary",
                                    #     key="download_prediction_output"
                                    # )
                                    
                                    # st.info("Scroll down to see visualization options!")
                                    return True  # Indicate success
                            else:
                                st.error("No results returned from the model.")
                                return False
                                
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON file: {e}")
                    return False
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    return False
            
            return False  # No results yet
    else:
        # No saved files found
        st.warning("No detection data found in current session.")
        st.info("You have two options:")
        st.markdown("1. **Go back and run Detection Probability Analysis first** (recommended)")
        st.markdown("2. **Upload all files manually** (fallback option)")
        
        choice = st.radio(
            "Choose your approach:",
            ["Auto-detect from scenario (search for files)", "Upload files manually"],
            key="no_saved_files_choice"
        )
        
        if choice == "Auto-detect from scenario (search for files)":
            st.info("Upload the prediction scenario - system will search for required files")
            
            uploaded_opt_json = st.file_uploader(
                "Upload Prediction Scenario", 
                type=["json"],
                help="System will search for muscat_input_var_core.json and detection files"
            )
            
            if uploaded_opt_json and st.button("Try Auto-Detection", type="primary", key="try_auto_detection"):
                try:
                    muscat_instance_json = json.load(uploaded_opt_json)
                    
                    with st.spinner("Searching for configuration and detection files..."):
                        results = run_model_with_auto_detection(muscat_instance_json)
                        
                        if results:
                            if "error" in results:
                                st.error(f"Auto-detection failed: {results['error']}")
                                st.info("**Solution**: Run Detection Probability Analysis first, or use manual upload below.")
                                return False
                            else:
                                # IMPORTANT: Store results in session state
                                st.session_state.model_results = results
                                st.session_state.model_completed = True
                                st.success("Found files and completed model successfully!")
                                
                                # Show download button for JSON results
                                results_json_string = json.dumps(results, indent=4)
                                instance_id = st.session_state.get('custom_instance_id', 'opt01')
                                output_filename = f"out_prediction_scenario_{instance_id}.json"
                                
                                st.download_button(
                                    label="Download Prediction Results JSON",
                                    data=results_json_string,
                                    file_name=output_filename,
                                    mime="application/json",
                                    type="primary",
                                    key="download_prediction_output_auto"
                                )
                                
                                st.info("Scroll down to see visualization options!")
                                return True  # Indicate success
                        else:
                            st.error("No results returned from the model.")
                            return False
                            
                except Exception as e:
                    st.error(f"Error: {e}")
                    return False
            
            st.markdown("---")
    
    # Manual upload option (always available as fallback)
    st.info("**Manual Upload Option**")
    st.markdown("Upload all required files manually if auto-detection doesn't work:")
    
    uploaded_opt_json = st.file_uploader("Upload Prediction Scenario", type=["json"], key="manual_pred")
    uploaded_var_core_json = st.file_uploader("Upload Configuration Scenario", type=["json"], key="manual_config")
    uploaded_detection_files = st.file_uploader(
        "Upload detection JSON files (multiple)", 
        type=["json"], 
        accept_multiple_files=True,
        key="manual_detection"
    )
    
    if uploaded_opt_json and uploaded_var_core_json and uploaded_detection_files:
        if st.button("Run Analytical Model (Manual Upload)", type="primary", key="run_model_manual"):
            try:
                muscat_instance_json = json.load(uploaded_opt_json)
                
                # Validate the uploaded instance
                validation_result = validate_sensor_keys(muscat_instance_json, uploaded_var_core_json)
                if not validation_result["valid"]:
                    st.error(f"Validation Error: {validation_result['error']}")
                    st.write("**Available sensor keys in configuration:**")
                    for key in validation_result.get("available_keys", []):
                        st.write(f"- `{key}`")
                    return False

                with st.spinner("Running analytical model..."):
                    results = run_analytical_model(
                        muscat_instance_json,
                        uploaded_var_core_json,
                        uploaded_detection_files
                    )
                
                if results:
                    if "error" in results:
                        st.error(f"Error: {results['error']}")
                        return False
                    else:
                        # IMPORTANT: Store results in session state
                        st.session_state.model_results = results
                        st.session_state.model_completed = True
                        st.success("Model completed successfully!")
                        
                        # Show download button for JSON results
                        results_json_string = json.dumps(results, indent=4)
                        instance_id = st.session_state.get('custom_instance_id', 'opt01')
                        output_filename = f"out_prediction_scenario_{instance_id}.json"
                        
                        st.download_button(
                            label=" Download Prediction Results JSON",
                            data=results_json_string,
                            file_name=output_filename,
                            mime="application/json",
                            type="primary",
                            key="download_prediction_output_manual"
                        )
                        
                        st.info(" Scroll down to see visualization options!")
                        return True  # Indicate success
                    
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                return False
            except Exception as e:
                st.error(f"Error running model: {e}")
                return False
    
    return False  # No results generated


def prediction_tab():
    """Main prediction tab with correct flow"""
    # Initialize session state
    initialize_prediction_session_state()
    
    st.header("Prediction - Custom Sensor Assignment")

    # Check session states
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

    # Map and assignment columns
    map_col, assignment_col = st.columns([3, 2])

    with map_col:
        st.subheader("Sensor Locations Map")
        m = folium.Map(location=st.session_state.map_center, zoom_start=14)

        # Draw area
        if st.session_state.boundary_type == "rectangle":
            folium.Rectangle(
                bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]],
                        [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                color='red',
                weight=2,
                fill=True,
                fill_color='red',
                fill_opacity=0.1
            ).add_to(m)
        elif st.session_state.boundary_type == "polygon":
            folium.Polygon(
                locations=st.session_state.boundary_points,
                color='red',
                weight=2,
                fill=True,
                fill_color='red',
                fill_opacity=0.1
            ).add_to(m)

        # Protected areas
        for i, area in enumerate(st.session_state.protected_areas):
            folium.Polygon(
                locations=area['points'],
                color='green',
                weight=2,
                fill=True,
                fill_color='green',
                fill_opacity=0.2,
                tooltip=area.get('name', f"Protected Area {i+1}")
            ).add_to(m)

        # Sensor locations
        for i, location in enumerate(st.session_state.potential_locations):
            sensor_name = location.get('name', f"Sensor {i+1}")
            has_assignment = sensor_name in st.session_state.prediction_sensor_assignments
            assigned_sensors = st.session_state.prediction_sensor_assignments.get(sensor_name, [])

            if has_assignment and assigned_sensors:
                marker_color = 'green'
                popup_text = f"{sensor_name}: {len(assigned_sensors)} sensor(s) assigned"
            else:
                marker_color = 'red'
                popup_text = f"{sensor_name}: No sensors assigned"

            folium.Marker(
                location=[location['lat'], location['lng']],
                popup=popup_text,
                tooltip=sensor_name,
                icon=folium.Icon(color=marker_color)
            ).add_to(m)

        folium_static(m, width=700, height=500)

    with assignment_col:
        st.subheader("Sensor Assignment")

        # Instance ID
        custom_id = st.text_input(
            "Enter Instance ID/Title:",
            value=st.session_state.custom_instance_id,
            key="custom_id_input"
        )
        st.session_state.custom_instance_id = custom_id

        # Location selection
        location_names = [loc.get('name', f"Sensor {i+1}") for i, loc in enumerate(st.session_state.potential_locations)]
        selected_location = st.selectbox("Select Location:", location_names)

        # Sensor selection
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

        # Generate JSON and download
        if st.session_state.prediction_sensor_assignments:
            muscat_instance = generate_muscat_binary_input()
            muscat_json_string = json.dumps(muscat_instance, indent=4)
            filename = f"prediction_scenario_{st.session_state.custom_instance_id}.json"

            st.download_button(
                label=f"Download Prediction Scenario",
                data=muscat_json_string,
                file_name=filename,
                mime="application/json",
                type="primary"
            )

    # Enhanced upload section - returns True if model completed successfully
    st.markdown("---")
    model_completed = enhanced_prediction_upload_section()
    
    # CRITICAL: Only show visualization AFTER model completes successfully
    if st.session_state.get('model_completed', False) and st.session_state.get('model_results'):
        st.markdown("---")
        #st.header("Results Visualization")
        #st.success("Model results are ready! Use the options below to visualize the data:")
        
        # Import and show visualization
        try:
            from prediction_results_visualizer import prediction_results_tab
            prediction_results_tab()
        except ImportError as e:
            st.error(f"Could not load visualization module: {e}")
        except Exception as e:
            st.error(f"Error in visualization: {e}")


# Rest of the helper functions remain the same...
def validate_sensor_keys(muscat_instance_json, var_core_json_file):
    """Validate that sensor keys in instance match those expected by the configuration"""
    try:
        # Reset file pointer if it's a file object
        if hasattr(var_core_json_file, 'seek'):
            var_core_json_file.seek(0)
            var_core_data = json.load(var_core_json_file)
            var_core_json_file.seek(0)  # Reset for later use
        else:
            # If it's already a dict
            var_core_data = var_core_json_file
        
        # Get available sensor keys from configuration
        available_sensors = var_core_data.get('sensors', {}).keys()
        
        # Get sensor keys from instance
        instance_sensors = muscat_instance_json.get('sensors', {}).keys()
        
        # Check if all instance sensors exist in configuration
        missing_sensors = set(instance_sensors) - set(available_sensors)
        
        if missing_sensors:
            return {
                'valid': False,
                'error': f"Sensor keys not found in configuration: {list(missing_sensors)}",
                'available_keys': list(available_sensors),
                'missing_keys': list(missing_sensors)
            }
        
        return {'valid': True}
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Validation error: {str(e)}"
        }


def generate_muscat_binary_input():
    """
    Generate MUSCAT binary input with exact sensor names matching detection_probability.py
    Now includes ALL available sensors (both assigned and unassigned)
    Unassigned sensors will have all zeros in their actual_locs arrays
    """
    if not st.session_state.get('sensor_specifications'):
        return {
            "sensors": {}, 
            "title": st.session_state.get('custom_instance_id', 'opt'),
            "config_file": "muscat_input_var_core.json"
        }

    total_locations = len(st.session_state.potential_locations)
    location_to_index = {loc.get('name', f"Sensor {i+1}"): i for i, loc in enumerate(st.session_state.potential_locations)}

    # Initialize sensor placements for ALL available sensors (with zeros)
    sensor_placements = {}
    
    # First, create entries for ALL sensor specifications with zeros
    for sensor_spec in st.session_state.sensor_specifications:
        exact_sensor_name = create_exact_sensor_name(sensor_spec)
        sensor_placements[exact_sensor_name] = [0] * total_locations

    # Then, update with actual assignments if any exist
    if st.session_state.get('prediction_sensor_assignments'):
        for location_name, assignments in st.session_state.prediction_sensor_assignments.items():
            for assignment in assignments:
                sensor_spec = assignment['spec']
                exact_sensor_name = create_exact_sensor_name(sensor_spec)
                
                # Set location assignment (this will override the zero)
                if location_name in location_to_index:
                    location_index = location_to_index[location_name]
                    sensor_placements[exact_sensor_name][location_index] = 1

    # Create sensors dictionary (now includes all sensors, assigned or not)
    sensors_dict = {key: {"actual_locs": val} for key, val in sensor_placements.items()}

    result = {
        "sensors": sensors_dict,
        "title": st.session_state.get('custom_instance_id', 'opt01'),
        "config_file": "muscat_input_var_core.json",
        "detection_files_pattern": "{sensor_name}_loc{location_id}_{uav_name}_detection.json"
    }
    
    return result


def run_analytical_model(muscat_instance_json, var_core_json_file, detection_files):
    """Run the analytical model with improved error handling"""
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}

    try:
        dgal.startDebug()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save var core JSON
            var_core_path = os.path.join(temp_dir, "muscat_input_var_core.json")
            
            # Handle file upload object vs file path
            if hasattr(var_core_json_file, 'read'):
                var_core_json_file.seek(0)
                with open(var_core_path, "wb") as f:
                    f.write(var_core_json_file.read())
            else:
                # If it's a file path string
                import shutil
                shutil.copy(var_core_json_file, var_core_path)

            # Save each detection file
            for uploaded_file in detection_files:
                uploaded_file.seek(0)  # Reset file pointer
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

            # Change directory so detection JSON files can be found
            old_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Load and validate configuration
                with open(var_core_path, 'r') as f:
                    muscatInputVarCore = json.load(f)

                # Verify inputs
                wr.verifyInputVar(muscatInputVarCore)
                wr.verifyInputInstance(muscat_instance_json, muscatInputVarCore)

                # Wrap inputs
                inputVarCore = wr.wrapInputVarCore(muscatInputVarCore)
                inputInstance = wr.wrapInputInstance(muscat_instance_json, inputVarCore)

                # Run sensor metrics
                output = sa.sensorMetrics(inputVarCore, inputInstance)
                
                return output

            finally:
                os.chdir(old_cwd)

    except FileNotFoundError as e:
        return {"error": f"File not found: {str(e)}", "type": "FileNotFoundError"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {str(e)}", "type": "JSONDecodeError"}
    except KeyError as e:
        return {"error": f"Missing required key in configuration: {str(e)}", "type": "KeyError"}
    except Exception as e:
        return {
            "error": str(e), 
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }