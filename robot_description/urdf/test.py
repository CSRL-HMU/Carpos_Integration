import numpy as np
np.int = int  # Συμβατότητα για την `urdfpy`
from urdfpy import URDF

# Φόρτωση URDF
robot = URDF.load("/home/soupia/catkin_ws/src/robot_description/urdf/husky_ur10.urdf")

# Εξαγωγή της κινηματικής αλυσίδας
base_link = "husky_base_link"
end_effector = "ur10_tool0"

chain = robot.get_chain(base_link, end_effector, links=True)
print(f"Αλυσίδα από {base_link} έως {end_effector}: {chain}")

# Υπολογισμός μήκους αλυσίδας
total_length = 0.0
for link in chain:
    if link.visual is not None and link.visual.origin is not None:
        origin = link.visual.origin
        total_length += origin[0]**2 + origin[1]**2 + origin[2]**2

print(f"Συνολικό μήκος αλυσίδας: {total_length}")




