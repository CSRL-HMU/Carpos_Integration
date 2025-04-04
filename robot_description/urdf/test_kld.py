from kdl_parser_py.urdf import treeFromParam
from PyKDL import ChainJntToJacSolver, JntArray

# Load tree from param server
(success, tree) = treeFromParam('/robot_description')
if not success:
    raise RuntimeError("Failed to extract kinematic tree from robot_description!")

chain = tree.getChain("husky_base_link", "ur10_tool0")
solver = ChainJntToJacSolver(chain)

# Define joint angles
joints = JntArray(chain.getNrOfJoints())
joints[0] = 0.0
joints[1] = 0.5

jacobian = solver.JntToJac(joints)
print(jacobian)
