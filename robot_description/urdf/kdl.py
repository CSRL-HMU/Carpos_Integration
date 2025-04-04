from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

urdf_path = "/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10_placeholderFIXED.urdf"

# Φόρτωση URDF
robot = URDF.from_xml_file(urdf_path)
success, tree = treeFromUrdfModel(robot)

if success:
    print("KDL Tree successfully created!")
    print(f"Number of segments: {tree.getNrOfSegments()}")
    print(f"Number of joints: {tree.getNrOfJoints()}")
    print(tree)
else:
    print("Failed to create KDL Tree from URDF")

