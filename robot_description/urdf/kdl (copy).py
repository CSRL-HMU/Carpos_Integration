import PyKDL
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import rospy
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import PyKDL as kdl

def parse_urdf(file_path):
    # Φόρτωση URDF αρχείου
    robot = URDF.from_xml_file(file_path)
    print(f"Robot name: {robot.name}")
    
    # Λίστα με όλα τα links
    print("\nLinks:")
    for link in robot.links:
        print(f"  - {link.name}")
    
    # Λίστα με όλα τα joints
    print("\nJoints:")
    for joint in robot.joints:
        print(f"  - {joint.name}: {joint.type}")
    
    # Δημιουργία KDL tree
    success, tree = treeFromUrdfModel(robot)
    if success:
        print("\nKDL Tree successfully created.")
        return robot, tree
    else:
        print("Failed to create KDL Tree.")
        return None, None

def calculate_jacobian(tree, base_link, end_effector, joint_angles):
    chain = tree.getChain(base_link, end_effector)
    jacobian = kdl.ChainJntToJacSolver(chain)
    joint_array = kdl.JntArray(len(joint_angles))
    
    for i, angle in enumerate(joint_angles):
        joint_array[i] = angle

    jac = kdl.Jacobian(len(joint_angles))
    jacobian.JntToJac(joint_array, jac)
    
    return jac

if __name__ == "__main__":
    rospy.init_node("urdf_parser", anonymous=True)

    # Path to URDF file
    urdf_file_path = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10.urdf"

    # Base link and end effector
    base_link = "husky_base_link"
    end_effector = "ur10_tool0"

    # Joint angles (αρχική θέση 0 για όλες τις αρθρώσεις)
    joint_angles = [0.0] * 6

    # Φόρτωση και εξαγωγή δεδομένων από το URDF
    robot, tree = parse_urdf(urdf_file_path)

    if tree:
        # Υπολογισμός Ιακωβιανής
        jacobian = calculate_jacobian(tree, base_link, end_effector, joint_angles)
        print("\nJacobian Matrix:")
        print(jacobian)

