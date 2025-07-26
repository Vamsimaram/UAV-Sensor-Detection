# test_dgal.py
import sys
sys.path.append("./aaa_lib_dgalPy")

try:
    import dgalPy as dgal
    print("✓ dgalPy imported successfully")
    
    # Test if pyomo is installed
    import pyomo.environ as pyo
    print("✓ Pyomo imported successfully")
    
    # Test other imports
    import sensorAssignmentModel as sa
    import muscat_wrappers as wr
    print("✓ All modules imported successfully")
    
    # Test dgal function
    dgal.startDebug()
    print("✓ dgalPy is working!")
    
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Install missing dependencies with: pip install pyomo")
except Exception as e:
    print(f"✗ Error: {e}")