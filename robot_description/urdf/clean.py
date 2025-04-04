import xml.etree.ElementTree as ET

def validate_and_fix_urdf(urdf_path, output_path, default_type="transmission_interface/SimpleTransmission"):
    """
    Ελέγχει και διορθώνει το URDF αρχείο προσθέτοντας τα απαραίτητα στοιχεία <type> στα transmission blocks.

    Args:
        urdf_path (str): Το αρχικό μονοπάτι του URDF αρχείου.
        output_path (str): Το μονοπάτι για το διορθωμένο URDF αρχείο.
        default_type (str): Default τύπος για το στοιχείο <type>.
    """
    # Φόρτωση URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Επεξεργασία όλων των transmission στοιχείων
    for transmission in root.findall("transmission"):
        type_elem = transmission.find("type")
        if type_elem is None:
            print(f"Προσθήκη <type> στο transmission: {transmission.attrib['name']}")
            new_type = ET.Element("type")
            new_type.text = default_type
            transmission.insert(0, new_type)  # Προσθήκη του type στοιχείου

    # Αποθήκευση του διορθωμένου URDF
    tree.write(output_path)
    print(f"Το URDF διορθώθηκε και αποθηκεύτηκε στο {output_path}")

# Παράδειγμα χρήσης
if __name__ == "__main__":
    urdf_file = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10.urdf"
    fixed_urdf_file = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_fixed.urdf"
    validate_and_fix_urdf(urdf_file, fixed_urdf_file)

