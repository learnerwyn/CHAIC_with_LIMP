"""Not the runnable version of video generation, further revising required."""

# import the json file including the TDW instructions
import json

with open('/put/your/path/here', 'r') as file:
    data = json.load(file)
print(type(data))

# using the dictionary data type as TDW controller command
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera

# set up a camera
camera = ThirdPersonCamera(position={"x": 2, "y": 1.6, "z": -0.6},
                           avatar_id="a",
                           look_at={"x": 0, "y": 0, "z": 0}) 

# communicate the json data with the TDW controller
c = Controller()
c.add_ons.extend([camera])
c.communicate(data)
c.communicate({"$type": "terminate"})