
import xml.etree.ElementTree as ET

def check_urdf_for_errors(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    errors = []
    for joint in root.findall("joint"):
        name = joint.get("name", "unknown")
        axis = joint.find("axis")
        origin = joint.find("origin")

        # Έλεγχος για ελλιπή axis
        if axis is None or "xyz" not in axis.attrib:
            errors.append(f"Joint '{name}' missing or invalid axis.")

        # Έλεγχος για ελλιπή origin
        if origin is None or not all(attr in origin.attrib for attr in ["xyz", "rpy"]):
            errors.append(f"Joint '{name}' missing or invalid origin.")

    return errors

# Χρήση
urdf_file = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_placeholderFIXED.urdf"
errors = check_urdf_for_errors(urdf_file)

if errors:
    print("Errors found in URDF:")
    for error in errors:
        print(f"- {error}")
else:
    print("No errors found in URDF.")
