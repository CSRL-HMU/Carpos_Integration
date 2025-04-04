import xml.etree.ElementTree as ET

def fix_urdf(file_path, output_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        # Προσθήκη origin αν λείπει
        if joint.find("origin") is None:
            origin = ET.SubElement(joint, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")

        # Προσθήκη axis αν δεν υπάρχει και η άρθρωση δεν είναι fixed
        if joint.get("type") != "fixed" and joint.find("axis") is None:
            axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1")

    # Αποθήκευση διορθωμένου URDF
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"URDF fixed and saved to {output_path}")

# Χρήση του script
input_file = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_placeholder.urdf"
output_file = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_placeholderFIXED.urdf"
fix_urdf(input_file, output_file)
