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
from typing import Dict, Tuple, Optional
from datetime import datetime

# -----------------------
# Imports & paths
# -----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "aaa_lib_dgalPy"))

try:
    import dgalPy as dgal  # noqa: F401  (import side-effects)
    import sensorAssignmentModel as sa
    import muscat_wrappers as wr
    print("Successfully imported dgalPy and dependencies.")
    DGALPY_AVAILABLE = True
except ImportError:
    traceback.print_exc()
    DGALPY_AVAILABLE = False
except Exception:
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


def sanitize_name(val: str) -> str:
    return (val or "").strip().replace(" ", "_").replace("-", "_").replace(".", "")


def create_exact_sensor_name(sensor_spec: dict) -> str:
    """Mirror the exact key emitted by detection step: Model_Manufacturer, with safety tweaks."""
    model = sanitize_name(sensor_spec.get("model") or "")
    manufacturer = sanitize_name(sensor_spec.get("manufacturer") or "")
    exact = f"{model}_{manufacturer}" if manufacturer else model
    if len(exact) > 50:
        exact = model
    if exact and exact[0].isdigit():
        exact = f"sensor_{exact}"
    return exact


# -----------------------
# var_core utilities (load + reconcile)
# -----------------------

def load_var_core_from_disk_or_session() -> Tuple[Optional[dict], Optional[str]]:
    """Returns (muscatInputVarCore, base_dir) if found, else (None, None)."""
    try:
        # 1) Session ZIP
        if 'zip_data' in st.session_state and st.session_state.zip_data:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(io.BytesIO(st.session_state.zip_data), 'r') as zf:
                    zf.extractall(temp_dir)
                # Be flexible with filenames
                names = set(os.listdir(temp_dir))
                core_name = None
                if "muscat_input_var_core.json" in names:
                    core_name = "muscat_input_var_core.json"
                elif "input_var_core.json" in names:
                    core_name = "input_var_core.json"
                else:
                    for n in names:
                        if n.endswith('.json') and 'muscat' in n:
                            core_name = n
                            break
                if core_name:
                    p = os.path.join(temp_dir, core_name)
                    with open(p, "r") as f:
                        return json.load(f), temp_dir
        # 2) On disk
        candidates = [
            os.getcwd(),
            os.path.join(os.getcwd(), "detection_outputs"),
            "/mnt/data",
        ]
        for d in candidates:
            p = os.path.join(d, "muscat_input_var_core.json")
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f), d
    except Exception:
        traceback.print_exc()
    return None, None


def var_core_location_lengths(muscatInputVarCore: dict) -> Dict[str, int]:
    """Return {exact_sensor_name: num_possible_locs} from var_core."""
    out: Dict[str, int] = {}
    sensors = (muscatInputVarCore or {}).get("sensors", {})
    for s_name, s_val in sensors.items():
        locs = (s_val or {}).get("possible_locs", []) or []
        out[s_name] = len(locs)
    return out


def reconcile_actual_locs_lengths(muscat_instance_json: dict, muscatInputVarCore: dict) -> dict:
    """Ensure each sensor's actual_locs matches var_core possible_locs length. Drops unknown sensors."""
    result = json.loads(json.dumps(muscat_instance_json))  # deep copy
    per_len = var_core_location_lengths(muscatInputVarCore)
    pruned_unknown = []
    for s_name, s_obj in list(result.get("sensors", {}).items()):
        want = per_len.get(s_name)
        if want is None:
            pruned_unknown.append(s_name)
            result["sensors"].pop(s_name, None)
            continue
        vec = (s_obj or {}).get("actual_locs", [])
        if len(vec) < want:
            vec = vec + [0] * (want - len(vec))
        elif len(vec) > want:
            vec = vec[:want]
        s_obj["actual_locs"] = vec
        result["sensors"][s_name] = s_obj
    if pruned_unknown:
        result["_debug_pruned_sensors"] = pruned_unknown
    return result


# -----------------------
# Scenario builder
# -----------------------

def generate_minimal_pred_scenario(instance_id: str) -> Tuple[dict, dict]:
    """Build the *minimal* prediction scenario expected by the analytical model."""
    title_id = instance_id or "opt01"

    var_core, _ = load_var_core_from_disk_or_session()
    per_sensor_len = var_core_location_lengths(var_core) if var_core else {}

    debug: Dict[str, object] = {"var_core_known": list(per_sensor_len.keys())}

    # If user hasn't loaded specs yet, return a skeleton scenario
    if not st.session_state.get('sensor_specifications'):
        return {
            "sensors": {},
            "title": title_id,
            "config_file": "muscat_input_var_core.json",
            "detec_file_pattern": "{sensor_name}_loc{location_id}_{uav_name}_detection.json",
            "id": title_id,
        }, debug

    # Build zero vectors per sensor, but ONLY for sensors present in var_core
    ui_loc_count = len(st.session_state.get('potential_locations', []))
    sensor_placements: Dict[str, list] = {}
    skipped = []
    for spec in st.session_state.sensor_specifications:
        exact = create_exact_sensor_name(spec)
        if var_core and exact not in per_sensor_len:
            skipped.append(exact)
            continue
        L = per_sensor_len.get(exact, ui_loc_count)
        sensor_placements[exact] = [0] * max(L, 0)

    debug["skipped_not_in_var_core"] = skipped

    # Map UI location names -> index in vector (1:1 with var_core order assumed)
    loc_names = [loc.get('name', f"Sensor {i+1}") for i, loc in enumerate(st.session_state.get('potential_locations', []))]
    name_to_idx = {n: i for i, n in enumerate(loc_names)}

    # Apply assignments
    for location_name, assignments in (st.session_state.get('prediction_sensor_assignments') or {}).items():
        if location_name not in name_to_idx:
            continue
        loc_idx = name_to_idx[location_name]
        for a in assignments:
            spec = a.get('spec') or {}
            exact = create_exact_sensor_name(spec)
            vec = sensor_placements.get(exact)
            if not vec:
                continue  # unknown or filtered out
            if 0 <= loc_idx < len(vec):
                vec[loc_idx] = 1

    scenario = {
        "sensors": {k: {"actual_locs": v} for k, v in sensor_placements.items()},
        "title": title_id,
        "config_file": "muscat_input_var_core.json",
        "detec_file_pattern": "{sensor_name}_loc{location_id}_{uav_name}_detection.json",
        "id": title_id,
    }
    return scenario, debug


# -----------------------
# Model runners (with better guards + messages)
# -----------------------

def _guard_after_reconcile(muscat_instance_json: dict, muscatInputVarCore: dict) -> Optional[str]:
    if not muscat_instance_json.get("sensors"):
        return (
            "No valid sensors remain after reconciling with muscat_input_var_core.json. "
            "This usually means the assigned UI sensor names don't exist in var_core."
        )
    has_any = any(any(v) for v in [s.get("actual_locs", []) for s in muscat_instance_json["sensors"].values()])
    if not has_any:
        return (
            "All sensors have empty/zero placement after reconcile. Assign at least one location that "
            "exists in var_core's possible_locs for that sensor."
        )
    return None


def run_model_with_memory_data(muscat_instance_json, memory_data):
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(io.BytesIO(memory_data["zip_data"]), 'r') as zf:
                zf.extractall(temp_dir)

            # Flexible core filename matching
            names = set(os.listdir(temp_dir))
            var_core_path = None
            for cand in ("muscat_input_var_core.json", "input_var_core.json"):
                if cand in names:
                    var_core_path = os.path.join(temp_dir, cand)
                    break
            if not var_core_path:
                for n in names:
                    if n.endswith('.json') and 'muscat' in n:
                        var_core_path = os.path.join(temp_dir, n)
                        break
            if not var_core_path:
                return {"error": "muscat_input_var_core.json not found in detection data"}

            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                with open(os.path.basename(var_core_path), 'r') as f:
                    var_core = json.load(f)

                muscat_instance_json = reconcile_actual_locs_lengths(muscat_instance_json, var_core)
                guard_msg = _guard_after_reconcile(muscat_instance_json, var_core)
                if guard_msg:
                    return {"error": guard_msg, "_debug": muscat_instance_json.get("_debug_pruned_sensors")}

                wr.verifyInputVar(var_core)
                wr.verifyInputInstance(muscat_instance_json, var_core)

                inputVarCore = wr.wrapInputVarCore(var_core)
                inputInstance = wr.wrapInputInstance(muscat_instance_json, inputVarCore)

                output = sa.sensorMetrics(inputVarCore, inputInstance)
                return output
            finally:
                os.chdir(old_cwd)
    except Exception as e:
        return {"error": f"Error running model with memory data: {str(e)}"}


def run_model_with_auto_detection(muscat_instance_json):
    if not DGALPY_AVAILABLE:
        return {"error": "dgalPy module not available"}

    # Prefer session ZIP produced by Detection tab or uploaded scenario ZIP
    if st.session_state.get('zip_data'):
        memory_data = {
            "zip_data": st.session_state.zip_data,
            # 'main_json' is optional; arrays and core are read from the ZIP
            "main_json": st.session_state.get('main_json'),
            "probability_files": st.session_state.get('probability_files', {})
        }
        return run_model_with_memory_data(muscat_instance_json, memory_data)

    # Fallback to disk
    try:
        config_file = muscat_instance_json.get("config_file", "muscat_input_var_core.json")
        candidate_paths = [
            config_file,
            os.path.join("detection_outputs", config_file),
            os.path.join(".", config_file),
            os.path.join("/mnt/data", config_file),
        ]
        var_core_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not var_core_path:
            return {"error": f"Configuration file '{config_file}' not found. Please run Detection Probability Analysis first."}

        old_cwd = os.getcwd()
        work_dir = os.path.dirname(var_core_path) or "."
        os.chdir(work_dir)
        try:
            with open(os.path.basename(var_core_path), 'r') as f:
                var_core = json.load(f)

            muscat_instance_json = reconcile_actual_locs_lengths(muscat_instance_json, var_core)
            guard_msg = _guard_after_reconcile(muscat_instance_json, var_core)
            if guard_msg:
                return {"error": guard_msg, "_debug": muscat_instance_json.get("_debug_pruned_sensors")}

            wr.verifyInputVar(var_core)
            wr.verifyInputInstance(muscat_instance_json, var_core)

            inputVarCore = wr.wrapInputVarCore(var_core)
            inputInstance = wr.wrapInputInstance(muscat_instance_json, inputVarCore)

            output = sa.sensorMetrics(inputVarCore, inputInstance)
            return output
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return {"error": f"Error running model with auto-detection: {str(e)}"}


def _autofill_from_uploaded_zip_if_needed():
    """
    If the user uploaded a detection ZIP (st.session_state.zip_data),
    populate session with potential_locations and sensor_specifications and persist
    parsed core/display so both Display and Prediction see the SAME data.
    """
    if not st.session_state.get("zip_data"):
        return

    need_locs = not st.session_state.get("potential_locations")
    need_specs = not st.session_state.get("sensor_specifications")
    need_bounds = not (st.session_state.get("sw_corner") and st.session_state.get("ne_corner"))

    try:
        with zipfile.ZipFile(io.BytesIO(st.session_state.zip_data), "r") as zf:
            names = set(zf.namelist())

            # ---- Robust core name detection
            core_name = None
            if "muscat_input_var_core.json" in names:
                core_name = "muscat_input_var_core.json"
            elif "input_var_core.json" in names:
                core_name = "input_var_core.json"
            else:
                for n in names:
                    if n.endswith(".json") and "muscat" in n and zf.getinfo(n).file_size > 0:
                        core_name = n
                        break

            core = None
            if core_name:
                core = json.loads(zf.read(core_name).decode("utf-8"))
                # Persist for parity with Detection/Display tabs
                st.session_state.main_json = json.dumps(core)
                st.session_state.muscat_input_var_core = core

            display = None
            if "display_map_data.json" in names:
                try:
                    display = json.loads(zf.read("display_map_data.json").decode("utf-8"))
                    st.session_state.display_map_data = display
                except Exception:
                    display = None

            # 1) potential_locations (prefer display names)
            if need_locs:
                filled = []
                # prefer display user_sensor_points
                if display and isinstance(display.get("user_sensor_points"), list):
                    for p in display["user_sensor_points"]:
                        lat = p.get("lat")
                        lng = p.get("lng", p.get("long", p.get("lon")))
                        if lat is None or lng is None:
                            continue
                        filled.append({
                            "name": p.get("name") or f"Sensor {len(filled)+1}",
                            "lat": float(lat),
                            "lng": float(lng),
                        })

                # fallback to core locations (unique by coords)
                if not filled and core and "sensors" in core:
                    idx = 1
                    seen = set()
                    for s in core["sensors"].values():
                        for loc in (s.get("possible_locs") or []):
                            lat = loc.get("lat")
                            lng = loc.get("lng", loc.get("long"))
                            if lat is None or lng is None:
                                continue
                            key = (round(float(lat), 6), round(float(lng), 6))
                            if key in seen:
                                continue
                            seen.add(key)
                            filled.append({
                                "name": f"sen_loc_{idx}",
                                "lat": float(lat),
                                "lng": float(lng),
                            })
                            idx += 1

                if filled:
                    st.session_state.potential_locations = filled

            # 2) sensor_specifications (lightweight from core)
            if need_specs and core and "sensors" in core:
                specs = []
                for s_name, s in core["sensors"].items():
                    specs.append({
                        "type": s.get("type", "sensor"),
                        "detection_range": s.get("detection_range", 0),
                        "response_time": s.get("detection_period", 0),
                        "model": s.get("model", s_name),
                        "manufacturer": s.get("make", ""),
                        "price_per_unit": s.get("ppu", 0),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                if specs:
                    st.session_state.sensor_specifications = specs

            # 3) bounds (AOI) from core or display
            if need_bounds:
                # from core.area_of_interest: list of [lon, lat]
                if core and isinstance(core.get("area_of_interest"), list) and core["area_of_interest"]:
                    lons = [float(p[0]) for p in core["area_of_interest"] if isinstance(p, (list, tuple)) and len(p) >= 2]
                    lats = [float(p[1]) for p in core["area_of_interest"] if isinstance(p, (list, tuple)) and len(p) >= 2]
                    if lats and lons:
                        st.session_state.sw_corner = (min(lats), min(lons))
                        st.session_state.ne_corner = (max(lats), max(lons))

                # display fallbacks
                if not (st.session_state.get("sw_corner") and st.session_state.get("ne_corner")) and display:
                    try:
                        # prefer explicit sw/ne if provided
                        if "sw" in display and "ne" in display:
                            swd = display["sw"]; ned = display["ne"]
                            sw_lon = float(swd.get("lon", swd.get("long")))
                            sw_lat = float(swd.get("lat"))
                            ne_lon = float(ned.get("lon", ned.get("long")))
                            ne_lat = float(ned.get("lat"))
                            st.session_state.sw_corner = (sw_lat, sw_lon)
                            st.session_state.ne_corner = (ne_lat, ne_lon)
                        else:
                            # generic polygon/box arrays (lon,lat)
                            candidates = []
                            for key in ("aoi_bounds", "aoi_polygon"):
                                if isinstance(display.get(key), list):
                                    candidates = display[key]
                                    break
                            if candidates:
                                lons = [float(p[0]) for p in candidates if isinstance(p, (list, tuple)) and len(p) >= 2]
                                lats = [float(p[1]) for p in candidates if isinstance(p, (list, tuple)) and len(p) >= 2]
                                if lats and lons:
                                    st.session_state.sw_corner = (min(lats), min(lons))
                                    st.session_state.ne_corner = (max(lats), max(lons))
                    except Exception:
                        pass
    except Exception:
        return


def _ensure_bounds_from_state_or_zip():
    """Populate st.session_state.sw_corner/ne_corner if missing."""
    if st.session_state.get("sw_corner") and st.session_state.get("ne_corner"):
        return

    pts = []
    if st.session_state.get("boundary_type") == "polygon" and st.session_state.get("boundary_points"):
        pts = st.session_state.boundary_points or []
    elif st.session_state.get("boundary_type") == "rectangle":
        pts = st.session_state.get("boundary_points") or []

    if pts:
        try:
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            st.session_state.sw_corner = (float(min(lats)), float(min(lons)))
            st.session_state.ne_corner = (float(max(lats)), float(max(lons)))
            return
        except Exception:
            pass

    # As a last resort, try to derive from uploaded ZIP again
    _autofill_from_uploaded_zip_if_needed()


def _normalize_protected_areas_in_session():
    """
    Normalize st.session_state.protected_areas into a list of dicts like:
      [{ "name": str, "points": [[lat, lon], ...] }, ...]
    Accepts various shapes/keys and lon/long/lng variations.
    """
    raw = st.session_state.get("protected_areas", [])
    parsed = []

    def to_latlon_pairs(seq):
        if not seq:
            return None
        a = seq[0]
        if not isinstance(a, (list, tuple)) or len(a) < 2:
            return None
        # heuristics: if first looks like lat,lon keep; else swap
        if abs(float(a[0])) <= 90:
            return [[float(p[0]), float(p[1])] for p in seq if isinstance(p, (list, tuple)) and len(p) >= 2]
        else:
            return [[float(p[1]), float(p[0])] for p in seq if isinstance(p, (list, tuple)) and len(p) >= 2]

    for idx, area in enumerate(raw):
        try:
            if isinstance(area, str):
                try:
                    area = json.loads(area)
                except Exception:
                    continue

            if isinstance(area, dict) and isinstance(area.get("points"), list):
                pts = to_latlon_pairs(area["points"])
                if pts and len(pts) >= 3:
                    parsed.append({"name": area.get("name", f"Protected Area {idx+1}"), "points": pts})
                continue

            if isinstance(area, dict) and ("geometry" in area or "type" in area):
                geom = area.get("geometry", area)
                gtype = geom.get("type")
                coords = geom.get("coordinates", [])
                ring = None
                if gtype == "Polygon":
                    ring = coords[0] if coords else []
                elif gtype == "MultiPolygon":
                    ring = coords[0][0] if coords and coords[0] else []
                if ring:
                    pts = to_latlon_pairs(ring)
                    if pts and len(pts) >= 3:
                        name = (area.get("properties") or {}).get("name") or area.get("name") or f"Protected Area {idx+1}"
                        parsed.append({"name": name, "points": pts})
                continue

            if isinstance(area, list) and area and isinstance(area[0], (list, tuple)):
                pts = to_latlon_pairs(area)
                if pts and len(pts) >= 3:
                    parsed.append({"name": f"Protected Area {idx+1}", "points": pts})
                continue

        except Exception:
            continue

    if parsed:
        st.session_state.protected_areas = parsed


# -----------------------
# Map viewport helpers
# -----------------------

def _compute_center_and_bounds():
    """
    Return (center_lat, center_lon, bounds) where bounds is [[sw_lat, sw_lon],[ne_lat, ne_lon]] or None.
    Priority:
      1) sw_corner/ne_corner in session
      2) boundary polygon/rectangle points
      3) potential_locations (mean)
      4) fallback to (20, 0)
    """
    sw = st.session_state.get("sw_corner")
    ne = st.session_state.get("ne_corner")
    if sw and ne:
        sw_lat, sw_lon = float(sw[0]), float(sw[1])
        ne_lat, ne_lon = float(ne[0]), float(ne[1])
        center_lat = (sw_lat + ne_lat) / 2.0
        center_lon = (sw_lon + ne_lon) / 2.0
        return center_lat, center_lon, [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # Try boundary points
    pts = st.session_state.get("boundary_points") or []
    if pts:
        lats = [float(p[0]) for p in pts]
        lons = [float(p[1]) for p in pts]
        sw_lat, sw_lon = min(lats), min(lons)
        ne_lat, ne_lon = max(lats), max(lons)
        center_lat = (sw_lat + ne_lat) / 2.0
        center_lon = (sw_lon + ne_lon) / 2.0
        return center_lat, center_lon, [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # Try potential sensor points
    locs = st.session_state.get("potential_locations") or []
    if locs:
        lats = [float(x["lat"]) for x in locs]
        lons = [float(x["lng"]) for x in locs]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        sw_lat, sw_lon = min(lats), min(lons)
        ne_lat, ne_lon = max(lats), max(lons)
        return center_lat, center_lon, [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # Fallback
    return 20.0, 0.0, None


def _expand_bounds(bounds, pad_ratio=0.25):
    """Pad bounds by a ratio to pull the camera back (leaflet fit)."""
    (sw_lat, sw_lon), (ne_lat, ne_lon) = bounds
    dlat = (ne_lat - sw_lat) or 1e-6
    dlon = (ne_lon - sw_lon) or 1e-6
    pad_lat = dlat * pad_ratio
    pad_lon = dlon * pad_ratio
    return [
        [sw_lat - pad_lat, sw_lon - pad_lon],
        [ne_lat + pad_lat, ne_lon + pad_lon],
    ]


# -----------------------
# UI: Prediction tab
# -----------------------

def prediction_tab():
    """Assign sensors, then one-click ZIP download: pred_scenario + out_pred_scen."""
    initialize_prediction_session_state()

    # Ensure any uploaded ZIP is parsed into session the same way Display does
    _autofill_from_uploaded_zip_if_needed()
    _ensure_bounds_from_state_or_zip()
    _normalize_protected_areas_in_session()

    st.header("Prediction - Custom Sensor Assignment")

    # Gate checks to keep user flow consistent with Detection tab
    if not st.session_state.get('location_selected', False):
        st.warning("Please select a location in the initial screen first.")
        if st.button("Return to Location Selection"):
            st.session_state.location_selected = False
            st.rerun()
        return

    if not st.session_state.get('area_selected', False):
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return

    if not st.session_state.get('potential_locations'):
        st.warning("Please place sensors in the Possible Sensor Placement tab first.")
        return

    if not st.session_state.get('sensor_specifications'):
        st.warning("Please add sensor specifications in the sidebar configuration first.")
        return

    # --- Compute center and padded bounds for the map view ---
    # Use AOI-based bounds first (so we zoom "from AOI rectangle onwards")
    disp = st.session_state.get("display_map_data") or {}
    core = st.session_state.get("muscat_input_var_core") or {}
    sqc = disp.get("square_coordinates") or core.get("square_coordinates")
    aoi_points = core.get("area_of_interest") or []  # [lon,lat]

    def _aoi_bounds_from(sqc, aoi_points):
        if isinstance(sqc, dict) and "sw" in sqc and "ne" in sqc:
            sw = sqc["sw"]; ne = sqc["ne"]
            return (float(sw[0]), float(sw[1])), (float(ne[0]), float(ne[1]))
        if aoi_points:
            lats = [float(lat) for lon, lat in aoi_points]
            lons = [float(lon) for lon, lat in aoi_points]
            return (min(lats), min(lons)), (max(lats), max(lons))
        return None

    aoi_bounds = _aoi_bounds_from(sqc, aoi_points)

    # Fallback center/bounds if AOI missing
    center_lat, center_lon, fallback_bounds = _compute_center_and_bounds()
    st.session_state.map_center = [center_lat, center_lon]

    # Helper to ensure we don't zoom to street level on tiny AOIs
    def _pad_bounds_min(bounds, min_deg=0.02, pad_ratio=0.25):
        (sw_lat, sw_lon), (ne_lat, ne_lon) = bounds
        dlat = ne_lat - sw_lat
        dlon = ne_lon - sw_lon
        # Enforce a minimum geographic extent
        if dlat < min_deg:
            pad = (min_deg - dlat) / 2.0
            sw_lat -= pad; ne_lat += pad
        if dlon < min_deg:
            pad = (min_deg - dlon) / 2.0
            sw_lon -= pad; ne_lon += pad
        # Add additional padding for aesthetics
        padded = _expand_bounds(((sw_lat, sw_lon), (ne_lat, ne_lon)), pad_ratio=pad_ratio)
        return padded

    # Choose which bounds to fit: AOI first, else fallback
    fit_bounds = aoi_bounds or fallback_bounds

    # Layout
    map_col, assignment_col = st.columns([3, 2])

    # Map
    with map_col:
        st.subheader("Sensor Locations Map")

        # Start moderately zoomed out; final view will be set by fit_bounds
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5, control_scale=True)

        # Fit to AOI (preferred) or fallback bounds with sane padding / min extent
        if fit_bounds:
            try:
                (sw_lat, sw_lon), (ne_lat, ne_lon) = fit_bounds
                # Larger padding for tiny AOIs so users see some context
                tiny = (abs(ne_lat - sw_lat) < 0.02) and (abs(ne_lon - sw_lon) < 0.02)
                padded = _pad_bounds_min(fit_bounds, min_deg=0.02, pad_ratio=0.35 if tiny else 0.25)
                m.fit_bounds(padded, padding=(40, 40))
            except Exception:
                pass

        # AOI rectangle (already computed from Display/Core)
        if isinstance(sqc, dict) and "sw" in sqc and "ne" in sqc:
            sw = sqc["sw"]; ne = sqc["ne"]
            folium.Rectangle(
                bounds=[[float(sw[0]), float(sw[1])], [float(ne[0]), float(ne[1])]],
                color='red', weight=2, fill=True, fill_color='red', fill_opacity=0.1
            ).add_to(m)
        elif aoi_points:
            lats = [float(lat) for lon, lat in aoi_points]
            lons = [float(lon) for lon, lat in aoi_points]
            folium.Rectangle(
                bounds=[[min(lats), min(lons)], [max(lats), max(lons)]],
                color='red', weight=2, fill=True, fill_color='red', fill_opacity=0.1
            ).add_to(m)

        # Protected areas overlay (already normalized elsewhere)
        disp = st.session_state.get("display_map_data") or {}
        core = st.session_state.get("muscat_input_var_core") or {}
        prot_polys = (disp.get("protected_areas_coordinates")
                      or core.get("protected_areas_coordinates")
                      or {})
        if not prot_polys and st.session_state.get('protected_areas'):
            prot_polys = {}
            for area in st.session_state.get('protected_areas', []):
                if isinstance(area, dict) and isinstance(area.get('points'), list):
                    prot_polys[area.get('name', f"Protected Area")] = area['points']
                elif isinstance(area, list):
                    prot_polys.setdefault("Protected Area", area)

        def _latlon_from_any(obj):
            if obj is None:
                return None, None
            if isinstance(obj, dict):
                if 'lat' in obj and any(k in obj for k in ('lon','long','lng')):
                    return float(obj['lat']), float(obj.get('lon', obj.get('long', obj.get('lng'))))
                if 'location' in obj and isinstance(obj['location'], (list, tuple)) and len(obj['location']) >= 2:
                    a, b = float(obj['location'][0]), float(obj['location'][1])
                    return (a, b) if abs(a) <= 90 else (b, a)
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                a, b = float(obj[0]), float(obj[1])
                return (a, b) if abs(a) <= 90 else (b, a)
            return None, None

        for name, coords in (prot_polys or {}).items():
            try:
                poly_pts = []
                for p in (coords or []):
                    lat, lon = _latlon_from_any(p)
                    if lat is not None and lon is not None:
                        poly_pts.append([lat, lon])
                if poly_pts:
                    if poly_pts[0] != poly_pts[-1]:
                        poly_pts.append(poly_pts[0])
                    folium.Polygon(
                        locations=poly_pts,
                        color='green', weight=2,
                        fill=True, fill_opacity=0.25,
                        tooltip=name,
                    ).add_to(m)
            except Exception as e:
                st.warning(f"Could not draw protected area '{name}': {e}")

        # Sensor markers
        for i, location in enumerate(st.session_state.potential_locations):
            sensor_name = location.get('name', f"Sensor {i+1}")
            assigned = st.session_state.prediction_sensor_assignments.get(sensor_name, [])
            marker_color = 'green' if assigned else 'red'
            popup_text = f"{sensor_name}: {len(assigned)} sensor(s) assigned" if assigned else f"{sensor_name}: No sensors assigned"
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

        # Scenario ID
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
                # prevent duplicates
                already = any(a['type'] == sensor_type and a['model'] == sensor_model
                              for a in st.session_state.prediction_sensor_assignments[selected_location])
                if not already:
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

        # If user has assigned at least one sensor, enable run & save
        if st.session_state.prediction_sensor_assignments:
            muscat_instance, debug_info = generate_minimal_pred_scenario(instance_id)

            # Run analytical model now so the ZIP contains both files
            results = run_model_with_auto_detection(muscat_instance)

            if isinstance(results, dict) and "error" in results:
                skipped = debug_info.get("skipped_not_in_var_core")
                if skipped:
                    st.error(
                        f"Model run failed: {results['error']}\n\n"
                        f"Skipped (not present in muscat_input_var_core.json): {skipped}"
                    )
                else:
                    st.error(f"Model run failed: {results['error']}")
            else:
                # success → keep results in state (optional for downstream display)
                st.session_state.model_results = results
                st.session_state.model_completed = True

                # Filter scenario to keys downstream expects
                allowed_keys = ["sensors", "title", "config_file", "detec_file_pattern", "id"]
                pred_scenario_filtered = {k: muscat_instance[k] for k in allowed_keys if k in muscat_instance}

                # Pretty strings
                pred_scenario_str = json.dumps(pred_scenario_filtered, indent=4)
                out_pred_str = json.dumps(results, indent=4)

                pred_fname = f"pred_scenario_{instance_id}.json"
                out_fname  = f"out_pred_scen_{instance_id}.json"
                zip_fname  = f"prediction_scenario_{instance_id}.zip"

                # Build a single ZIP (prediction + detection artifacts if available)
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(pred_fname, pred_scenario_str)
                    zf.writestr(out_fname, out_pred_str)

                    try:
                        if st.session_state.get("zip_data"):
                            with zipfile.ZipFile(io.BytesIO(st.session_state.zip_data), 'r') as det_zip:
                                existing = set(zf.namelist())
                                for member in det_zip.namelist():
                                    if member not in existing:
                                        zf.writestr(member, det_zip.read(member))
                        elif st.session_state.get("detection_zip_path"):
                            det_path = st.session_state.detection_zip_path
                            if det_path and os.path.exists(det_path):
                                with zipfile.ZipFile(det_path, 'r') as det_zip:
                                    existing = set(zf.namelist())
                                    for member in det_zip.namelist():
                                        if member not in existing:
                                            zf.writestr(member, det_zip.read(member))
                    except Exception as e:
                        st.warning(f"Couldn't merge detection files into the scenario ZIP: {e}")

                st.download_button(
                    label="Save Scenario",
                    data=zip_buf.getvalue(),
                    file_name=zip_fname,
                    mime="application/zip",
                    key=f"dl_zip_only_{instance_id}"
                )

        else:
            st.info(
                "No sensors assigned yet. Select a location and add at least one sensor "
                "in the **Sensor Assignment** panel above."
            )
