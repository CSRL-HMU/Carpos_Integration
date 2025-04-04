import xml.etree.ElementTree as ET

def replace_transmissions_with_placeholders(urdf_input_path, urdf_output_path):
    # Φορτώστε το URDF ως XML δέντρο
    tree = ET.parse(urdf_input_path)
    root = tree.getroot()

    # Εντοπίστε όλα τα transmission tags
    transmissions = root.findall("transmission")
    for transmission in transmissions:
        # Αποθηκεύστε το όνομα του transmission
        transmission_name = transmission.get("name")
        # Διαγράψτε το παλιό transmission
        parent = root
        parent.remove(transmission)

        # Δημιουργήστε placeholder για transmission
        placeholder = ET.Element("transmission", {"name": transmission_name})
        type_tag = ET.SubElement(placeholder, "type")
        type_tag.text = "transmission_interface/SimpleTransmission"

        # Προσθέστε joint και actuator
        joint_tag = ET.SubElement(placeholder, "joint", {"name": transmission_name.replace("_trans", "")})
        joint_type = ET.SubElement(joint_tag, "hardwareInterface")
        joint_type.text = "hardware_interface/EffortJointInterface"

        actuator_tag = ET.SubElement(placeholder, "actuator", {"name": f"{transmission_name}_actuator"})
        mechanical_reduction = ET.SubElement(actuator_tag, "mechanicalReduction")
        mechanical_reduction.text = "1.0"

        # Εισαγωγή του placeholder στο δέντρο
        root.append(placeholder)

    # Γράψτε το νέο URDF στο αρχείο εξόδου
    tree.write(urdf_output_path, encoding="utf-8", xml_declaration=True)
    print(f"URDF updated with placeholder transmissions: {urdf_output_path}")

# Χρήση
if __name__ == "__main__":
    input_urdf = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10.urdf"
    output_urdf = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_placeholder.urdf"
    replace_transmissions_with_placeholders(input_urdf, output_urdf)
