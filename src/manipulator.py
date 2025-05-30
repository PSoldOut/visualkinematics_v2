from __future__ import annotations
import numpy as np
import pythreejs as three
from ipywidgets import *
from IPython.display import display
from pythreejs import *
import time
from scipy.spatial.transform import Rotation as R, Slerp
import manager
import util
from threading import Timer
from numba import njit
from typing import *
import re

pending_actions = []
max_pending_actions = 100





def block(data, current:Kinematic_Chain_Element, func:Callable, depth:int = 0) -> None:
    with Widget.hold_sync(current.get_renderable()):
        if len(current.children) > 0:
            for child in current.children:
                block(data, child, func, depth+1)
        else:
            func(data)


def add_pending_action(action:AnimationAction) -> None:
    global pending_actions
    pending_actions.append(action)
    display(f"pending actions: {len(pending_actions)}")
    if len(pending_actions) > max_pending_actions:
        for action in pending_actions:
            action.stop()
        pending_actions = []


def pose_to_matrix(xyz:np.ndarray, rpy:np.ndarray, degrees:bool = True) -> np.ndarray:
    r = R.from_euler('ZYX', rpy, degrees=degrees)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = xyz
    return T



def compute_dh_matrix(theta:float, d:float, a:float, alpha:float) -> np.ndarray:
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,      ca,     d],
        [0,        0,       0,     1]
    ])




#-----------------------------------------------------------------------------------------------------------------------------------------------




class DHKinematicModel:
    def __init__(self, dh_parameters:dict):  # dh_parameters ist dict von dicts mit theta, d, a, alpha
        self.dh_parameters:dict = dh_parameters
        self.joint_angles:dict = {}
        for name, _ in dh_parameters.items():
            self.joint_angles[name] = 0.0

    def compute_transforms(self) -> dict: 
        T_dict:dict = {}
        T:np.ndarray = np.eye(4)
        for name, param in self.dh_parameters.items():
            theta = self.joint_angles[name] + param["theta"]
            d = param['d']
            alpha = param['alpha']
            a = param['a']
            T_i = compute_dh_matrix(theta, d, a, alpha)
            T = T @ T_i
            T_dict[name] = T
        return T_dict
    

    def update_joint_angle(self, name:str, angle_rad:float) -> None:
        self.joint_angles[name] = angle_rad


    def compute_dh_matrix(self, theta:float, d:float, a:float, alpha:float) -> np.ndarray:
        return compute_dh_matrix(theta, d, a, alpha)








#----------------------------------------------------------------------------------------------------------------







class Kinematic_Chain_Element:
    def __init__(self, name : str):
        self.name:str = name
        self.children:list[Kinematic_Chain_Element] = []
        self.parent:Kinematic_Chain_Element|None = None
        self.renderable:Object3D = three.Group()
        self.frame:Object3D = util.create_axes(0.3, show_labels=False, arrow_size=0.1, transparent_arrows=False)
        self.frame.visible = False


    def get_renderable(self) -> Object3D:
        return self.renderable

    def add(self, object:Object3D|Kinematic_Chain_Element) -> None:
        renderable = object
        if hasattr(object, "get_renderable"):
            renderable = object.get_renderable()
        self.renderable.add(renderable)
        self.children.append(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = self

    def get_child_by_name(self, name:str) -> Kinematic_Chain_Element|None:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def get_matching_children(self, regex:str) -> list[Kinematic_Chain_Element]:
        result:list[Kinematic_Chain_Element] = []
        for c in self.children:
            if re.fullmatch(regex, c.name):
                result.append(c)
        return result

    def remove(self, object) -> None:
        self.children.remove(object)
        if hasattr(object, "get_renderable"):
            self.renderable.remove(object.get_renderable())
        elif isinstance(object, Object3D):
            self.renderable.remove(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = None


    def show_frame(self, show:bool = True) -> None:
        self.frame.visible=show

    def set_position(self, vec:np.ndarray):
        util.set_translation(self.renderable, vec)

    def get_position(self):
        return self.renderable.position

    def set_rotation(self, vec_degree):
        util.set_rotation(self.renderable, vec_degree, "ZYX")

    def get_rotation(self, degrees=True):
        q = self.renderable.quaternion  # Quaternion: [x, y, z, w]
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        euler_deg = r.as_euler("ZYX", degrees=degrees)
        return euler_deg.tolist()
    

    def get_rotation_as_quaternion(self):
        return list(self.get_renderable().quaternion)











#----------------------------------------------------------------------------------------------------------------------







class Tool(Kinematic_Chain_Element):
    def __init__(self, name:str):
        super().__init__(name)
        self.links:list[Link] = []
        self.joints:list[Joint] = []
        xacro_filepath:str = manager.find_xacro_filepath_by_robot_name(name)
        urdf:str = manager.xacro_to_urdf_string(xacro_filepath)
        self.urdf_dictionary:dict = manager.parse_urdf(urdf)

        #self.urdf_dictionary = json.dumps(self.urdf_dictionary, indent=4)
        #print(self.urdf_dictionary)
        #self.urdf_dictionary = json.loads(self.urdf_dictionary)
        
        self.init_links()
        self.init_joints()
        self.renderable = self.links[0].get_renderable()
        

        self.correct_elements()
        


    def correct_elements(self):
        #nicht schön aber notwendig
        if self.name == "robotiq_arg2f_140_model":
            order = "xyz"
            current = None

            current = self.get_joint_by_name("finger_joint")
            util.set_rotation(current.get_renderable(), [-70,180,180],order)

            current = self.get_joint_by_name("left_outer_finger_joint")
            util.set_rotation(current.get_renderable(), [0,0,0],order)

            current = self.get_joint_by_name("left_inner_knuckle_joint")
            util.set_rotation(current.get_renderable(), [70,0,180],order)

            current = self.get_joint_by_name("left_inner_finger_joint")
            util.set_rotation(current.get_renderable(), [-20,0,0],order)

            current = self.get_joint_by_name("left_inner_finger_pad_joint")
            util.set_rotation(current.get_renderable(), [0,0,0],order)

            current = self.get_joint_by_name("right_outer_knuckle_joint")
            util.set_rotation(current.get_renderable(), [110,0,-180],order)

            current = self.get_joint_by_name("right_outer_finger_joint")
            util.set_rotation(current.get_renderable(), [0,0,0],order)

            current = self.get_joint_by_name("right_inner_knuckle_joint")
            util.set_rotation(current.get_renderable(), [110,0,-180],order)

            current = self.get_joint_by_name("right_inner_finger_joint")
            util.set_rotation(current.get_renderable(), [-20,0,0],order)

            current = self.get_joint_by_name("right_inner_finger_pad_joint")
            util.set_rotation(current.get_renderable(), [0,0,0],order)



    def set_opacity(self, opacity:float):
        for c in self.links:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity
        for c in self.joints:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity


    def get_link_by_name(self, name:str) -> Link:
        for l in self.links:
            if l.name == name:
                return l
        return None

    def get_joint_by_name(self, name:str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        return None
    

    def show_joint_frames(self, show=True):
        for j in self.joints:
            j.frame.visible=show

    def show_link_frames(self, show=True):
        for l in self.links:
            l.frame.visible=show


    def init_links(self):
        scale = None
        for link_element in self.urdf_dictionary["links"]:
            meshi = None
            xyz = [0,0,0]
            rpy = [0,0,0]
            if len(link_element["visual"]) > 0:
                if link_element["visual"][0]["origin"] is not None:
                    pass
                    xyz = link_element["visual"][0]["origin"]["xyz"].split()
                    xyz = [float(x) for x in xyz]
                    rpy = link_element["visual"][0]["origin"]["rpy"].split()
                    rpy = [np.rad2deg(float(x)) for x in rpy]   
                if "filename" in link_element["visual"][0]["geometry"] and link_element["visual"][0]["geometry"]["filename"] is not None:             
                    mesh_path = link_element["visual"][0]["geometry"]["filename"]
                if "scale" in link_element["visual"][0]["geometry"] and link_element["visual"][0]["geometry"]["scale"] is not None and scale is None:
                    scale = link_element["visual"][0]["geometry"]["scale"].split()
                    scale = [float(x) for x in scale]
                if link_element["visual"][0]["material"] is not None:
                    material_name : str = link_element["visual"][0]["material"]["name"]
                    rgba = None
                    if material_name == "":
                        rgba : str = link_element["visual"][0]["material"]["rgba"].split()
                        rgba = [float(x) for x in rgba]
                        hex_color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                        if mesh_path is None:
                            t = link_element["visual"][0]["geometry"]["type"]
                            size = link_element["visual"][0]["geometry"]["params"]["size"].split()
                            size = [float(x) for x in size]
                            if t == "box":
                                meshi = Mesh(
                                BoxBufferGeometry(size[0], size[1], size[2]),
                                MeshStandardMaterial(color=hex_color, opacity=rgba[3], transparent=True),
                                position=[2, 0, 4]
)
                        else:
                            meshi = manager.load_mesh_auto_compatibility(mesh_path, hex_color, rgba[3])
                    else:
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, link_element["visual"][0]["material"]["name"])
                else:
                    meshi = manager.load_mesh_auto_compatibility(mesh_path)
            else:
                meshi = three.Group()
            if scale is not None:
                meshi.scale=scale
                scale = [1,1,1]
            l = Link(link_element["name"], meshi, xyz, rpy)
            self.links.append(l)





    def init_joints(self):
        for i in range(len(self.urdf_dictionary["joints"])):
            joint_element = self.urdf_dictionary["joints"][i]
            pos = joint_element["origin"]["xyz"].split()
            pos = [float(x) for x in pos]
            angles = joint_element["origin"]["rpy"].split()
            angles = [np.rad2deg(float(x)) for x in angles]
            joint_parent = self.get_link_by_name(joint_element["parent"])
            joint_child = self.get_link_by_name(joint_element["child"])
            joint_axis = joint_element["axis"]
            if joint_axis is not None:
                joint_axis = joint_axis["xyz"].split()
                joint_axis = [float(x) for x in joint_axis]
            joint : Joint = Joint(joint_element["name"], np.array(joint_axis), np.array(pos), np.array(angles))
            joint.add(joint_child)
            joint_parent.add(joint) 
            if joint_element["mimic"] is not None:
                multiplier:float = 1
                if "multiplier" in joint_element["mimic"] and joint_element["mimic"]["multiplier"] is not None:
                    multiplier:float = float(joint_element["mimic"]["multiplier"])
                gets_mimiced:Joint = self.get_joint_by_name(joint_element["mimic"]["joint"])
                gets_mimiced.add_mimicer((joint, multiplier))
            self.joints.append(joint)

        











#-----------------------------------------------------------------------------------------------------------------------












class Joint(Kinematic_Chain_Element):
    def __init__(self, name:str, axis:np.ndarray, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0])):
        super().__init__(name)
        self.renderable : Object3D = three.Group()
        self.axis:np.ndarray = axis
        self.mimicers:list[tuple[Kinematic_Chain_Element, float]] = []
        self.renderable.add(self.frame)
        self.set_position(position)
        self.set_rotation(rotation)

        self.dh_alignment:np.ndarray = np.eye(4)  # später ggf. mit echten Werten setzen

    def add_mimicer(self, mimicer_and_multiplier:tuple[Kinematic_Chain_Element, float]):
        self.mimicers.append(mimicer_and_multiplier)
    





class Link(Kinematic_Chain_Element):
    def __init__(self, name:str, mesh:Mesh, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0])) -> None:
        super().__init__(name)
        self.renderable:Mesh = mesh
        self.renderable.add(self.frame)
        self.set_position(position)
        self.set_rotation(rotation)
        
    

        










#----------------------------------------------------------------------------------------------------------------------









class Manipulator:
    def __init__(self, name:str, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        self.name:str = name
        self.links:list[Link] = []
        self.joints:list[Joint] = []
        self.dh_model:DHKinematicModel|None = None
        self.base_to_dh:np.ndarray = np.eye(4)
        self.dh_to_tool:np.ndarray = np.eye(4)
        xacro_filepath:str = manager.find_xacro_filepath_by_robot_name(name)
        urdf:str = manager.xacro_to_urdf_string(xacro_filepath)
        self.urdf_dictionary:dict = manager.parse_urdf(urdf)
        self.tool:Tool|None = None
        if tool_name!="":
            self.tool = Tool(tool_name)
        
        #self.urdf_dictionary = json.dumps(self.urdf_dictionary, indent=4)
        #print(self.urdf_dictionary)
        #self.urdf_dictionary = json.loads(self.urdf_dictionary)

        self.init_links()
        self.init_joints()

        self.base_link:Link = self.get_link_by_name("base_link")
        if self.base_link is None: raise AssertionError("Baselink wurde nicht gefunden")

        self.base_link_to_link_1_joint:Joint = self.find_base_link_to_link_1_joint()

        self.base:Link|None = self.get_link_by_name("base")
        if self.base is None: raise AssertionError("Base wurde nicht gefunden")

        self.mesh:Object3D = self.base_link.get_renderable()
        self.flange:Link = self.find_flange()
        self.link_to_flange_joint:Joint = self.flange.parent

        
        self.flange_to_tool_joint:Joint = self.find_flange_to_tool_joint()

        self.base_to_baselink_joint:Joint = self.base.parent
        self.baselink_to_base_joint:Joint = self.base.parent
        if tool_name!="":
            t = self.get_link_by_name("tool0")
            t.add(self.tool)
        
        
    def compute_base_to_dh_transformation():
        pass
    def compute_dh_to_tool_transformation():
        pass

    def get_global_tcp_position(self):
        trans = self.dh.compute_transforms()
        return trans["tool0"]


    def animate_stable(self, joints:list, angles_rad:list, duration:float = 2) -> None:
        action_q2_joint:list = []
        for joint, angle_rad in zip(joints, angles_rad):
            action_q2_joint.append(apply_joint_rotation_animated(joint=joint, axis=joint.axis, angle_rad=angle_rad, duration=duration))

        print(len(action_q2_joint))
        def on_animation_finished():
            #display("DRINNE")
            base = joints[0]
            while(base.parent is not None):
                base = base.parent


            def do(action_q2_joint_array):
                base.get_renderable().visible = False
                time.sleep(0.02)
                for i in range(len(action_q2_joint)):
                    action_q2_joint[i][2].get_renderable().quaternion = tuple(action_q2_joint[i][1])
                    #action_q2_joint[i][0].stop()
                    action_q2_joint[i][2].get_renderable().quaternion = tuple(action_q2_joint[i][1])
                base.get_renderable().visible = True
                    #display("jauuuuuu")


            if action_q2_joint[0][0].time <= 0.000001:
                block([action_q2_joint], base, do)
                #display("animation finished!")
            else:
                i = 0
                while(action_q2_joint[0][0].time > 0.000001 and i < 99):
                    time.sleep(0.001)
                    i+=1
                block(action_q2_joint, base, do)
                #display("animation finished!")
        Timer(duration, on_animation_finished).start()



        
    def animate_experimental(self, joints:list, angles_rad:list, duration:float = 1.0, quality:float = 1):

        def add_mimicers(current:Joint, current_angle_rad, joints:list):
            joints.append(current)
            angles_rad.append(current_angle_rad)
            for m in current.mimicers:
                add_mimicers(current=m[0], current_angle_rad=m[1] * current_angle_rad, joints=joints)

        slerps:list = []

        for j, angle in zip(joints, angles_rad):
            for m in j.mimicers:
                add_mimicers(current=m[0], current_angle_rad=m[1]*angle, joints=joints)
    
        for joint, angle_rad in zip(joints, angles_rad):
            axis = joint.axis
            axis = np.array(axis, dtype=np.float64)
            if np.linalg.norm(axis) == 0:
                raise ValueError("Rotationsachse darf nicht der Nullvektor sein.")
            axis = axis / np.linalg.norm(axis)
            base_rot = R.from_quat(joint.get_rotation_as_quaternion())
            #base_rot = R.from_euler("ZYX", joint.get_rotation(), degrees=True)
            axis_rot = R.from_rotvec(axis * angle_rad)
            final_rot = axis_rot * base_rot
            #q1s.append(list(joint.get_renderable().quaternion))
            #q2s.append(final_rot.as_quat())
            q1 = list(joint.get_renderable().quaternion)
            q2 = final_rot.as_quat()
            key_times = np.array([0, 1])  
            key_rots = R.from_quat([q1, q2]) 
            slerp = Slerp(key_times, key_rots)
            slerps.append(slerp)
        
        t = 0




        def step(data):
            for joint, slerp, angle_rad in zip(joints, slerps, angles_rad):
                joint.get_renderable().quaternion = tuple(slerp(data).as_quat())
                

        while(t < duration):
            start = time.perf_counter()
            block(t/duration, self.base_link, step)
            time.sleep(0.01/quality)
            end = time.perf_counter()
            t += end - start
        block(1, self.base_link, step)



    def apply_DH_model(self, dh:DHKinematicModel) -> None:
        self.dh = dh
        DH_transforms:dict = dh.compute_transforms()
        global_transforms:dict = self.compute_global_transform()
        for name, transform in DH_transforms.items():
            joint:Joint = self.get_joint_by_name(name)
            joint.dh_alignment = np.linalg.inv(global_transforms[name]) @ transform
        

        
        self.dh_to_tool = global_transforms[self.flange_to_tool_joint.name] @ np.linalg.inv(DH_transforms[self.link_to_flange_joint.name]) 

        base_transform:np.ndarray = pose_to_matrix(self.base_to_baselink_joint.get_position(), self.base_to_baselink_joint.get_rotation(False), False)
        self.base_to_dh = DH_transforms[self.base_link_to_link_1_joint.name] @ np.linalg.inv(base_transform)
        




    

    def init_links(self) -> None:
        for link_element in self.urdf_dictionary["links"]:
            meshi = None
            xyz = [0,0,0]
            rpy = [0,0,0]
            if len(link_element["visual"]) > 0:
                if link_element["visual"][0]["origin"] is not None:
                    pass
                    xyz = link_element["visual"][0]["origin"]["xyz"].split()
                    xyz = [float(x) for x in xyz]
                    rpy = link_element["visual"][0]["origin"]["rpy"].split()
                    rpy = [np.rad2deg(float(x)) for x in rpy]                
                mesh_path = link_element["visual"][0]["geometry"]["filename"]
                if link_element["visual"][0]["material"] is not None:
                    material_name : str = link_element["visual"][0]["material"]["name"]
                    rgba = None
                    if material_name == "":
                        rgba : str = link_element["visual"][0]["material"]["rgba"].split()
                        rgba = [float(x) for x in rgba]
                        hex_color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, hex_color, rgba[3])
                    else:
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, link_element["visual"][0]["material"]["name"])
                else:
                    meshi = manager.load_mesh_auto_compatibility(mesh_path)
            else:
                meshi = three.Group()
            l = Link(link_element["name"], meshi, xyz, rpy)
            self.links.append(l)





    def init_joints(self) -> None:
        for i in range(len(self.urdf_dictionary["joints"])):
            joint_element = self.urdf_dictionary["joints"][i]
            pos = joint_element["origin"]["xyz"].split()
            pos = [float(x) for x in pos]
            angles = joint_element["origin"]["rpy"].split()
            angles = [np.rad2deg(float(x)) for x in angles]
            joint_parent = self.get_link_by_name(joint_element["parent"])
            joint_child = self.get_link_by_name(joint_element["child"])
            joint_axis = joint_element["axis"]
            if joint_axis is not None:
                joint_axis = joint_axis["xyz"].split()
                joint_axis = [float(x) for x in joint_axis]
            joint:Joint = Joint(joint_element["name"], joint_axis, pos, angles)
            joint.add(joint_child)
            joint_parent.add(joint) 
            if joint_element["mimic"] is not None:
                multiplier:float = 1
                if "multiplier" in joint_element["mimic"] and joint_element["mimic"]["multiplier"] is not None:
                    multiplier:float = float(joint_element["mimic"]["multiplier"])
                gets_mimiced:Joint = self.get_joint_by_name(joint_element["mimic"]["joint"])
                gets_mimiced.add_mimicer([joint, multiplier])
            self.joints.append(joint)



    def show_joint_frames(self, show=True):
        for j in self.joints:
            j.frame.visible=show

    def show_link_frames(self, show=True):
        for l in self.links:
            l.frame.visible=show

    def show_DH_frames(self, show=True):
        current = self.base_link
        if isinstance(current, Joint):
            current.frame.visible=show
            util.apply_transformation_matrix(current.frame, current.dh_alignment)
        while(len(current.children) > 0):
            current = current.children[0]
            if isinstance(current, Joint):
                current.frame.visible=show
                util.apply_transformation_matrix(current.frame, current.dh_alignment)

    def get_renderable(self):
        return self.mesh
    
    def get_link_by_name(self, name:str) -> Link|None:
        for l in self.links:
            if l.name == name:
                return l
        return None

    def get_joint_by_name(self, name:str) -> Joint|None:
        for j in self.joints:
            if j.name == name:
                return j
        return None
    



    def get_matching_joints(self, regex:str) -> list[Joint]:
        result : list[Joint] = []
        for j in self.joints:
            if re.fullmatch(regex, j.name):
                result.append(j)
        return result
    

    def get_matching_links(self, regex:str) -> list[Link]:
        result : list[Link] = []
        for l in self.links:
            if re.fullmatch(regex, l.name):
                result.append(l)
        return result

    def find_flange(self) -> Link:
        arr:list[Link]= []
        for l in self.links:
            if "flange" in l.name and "tool0" not in l.name:
                arr.append(l)
        if len(arr) != 1:
            raise AssertionError("mehr als zwei flange-links gefunden")
        return arr[0]
        

    def find_flange_to_tool_joint(self) -> Joint:
        matching_flange_children = self.find_flange().get_matching_children(".*tool0.*")
        if len(matching_flange_children)!=1:
            raise AssertionError(f"Es wurden {len(matching_flange_children)} kinder vom Flange gefunden statt nur eines.")
        elif not isinstance(matching_flange_children[0], Joint):
            raise TypeError(f"joint zwischen flange und tool nicht gefunden. stadessen wurde {matching_flange_children[0].name} gefunden")
        return matching_flange_children[0]

    def find_base_link_to_link_1_joint(self) -> Joint:
        matching_base_link_children = self.base_link.get_matching_children(".*joint.*1.*")
        if len(matching_base_link_children) != 1:
            raise AssertionError("es wurde mehr als ein joint vom base_link zu link_1 gefunden")
        elif not isinstance(matching_base_link_children[0], Joint):
            raise TypeError(f"statt joint eins wurde hier {matching_base_link_children[0].name} gefunden von der Klasse {matching_base_link_children[0].__class__}")
        return matching_base_link_children[0]

    def set_opacity(self, opacity:float, tool_also:bool = False) -> None:
        for c in self.links:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity
        for c in self.joints:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity
        if tool_also:
            self.tool.set_opacity(opacity)


    def print_links(self) -> None:
        print("\nLinks:")
        for link in self.links:
            print(link.name)
        print()

    def print_joints(self) -> None:
        print("\nJoints:")
        for joint in self.joints:
            print(joint.name)
        print()

    def print_kinematic_chain(self) -> None:
        print("\nKinematic Chain:")
        current = self.links[0]
        print(current.name)
        while(len(current.children) > 0):
            current = current.children[0]
            print(current.name)
        print()


    def compute_global_transform(self, current:Kinematic_Chain_Element = None, parent_transform:np.ndarray = np.eye(4), global_transforms:dict = None, with_print:bool = False):
        if global_transforms is None:
            global_transforms = {}
        if current is None:
            current:Kinematic_Chain_Element = self.base_link
        current_transform = parent_transform @ pose_to_matrix(current.get_position(), current.get_rotation(False), False)
        global_transforms[current.name] = current_transform.copy()
        if with_print:
            print(current.name,":  pose:", current.get_position(), " , ", current.get_rotation(False), " becomes to: ")
            print(current_transform)
        for child in current.children:
            self.compute_global_transform(child, current_transform, global_transforms)
        return global_transforms












#-------------------------------------------------------------------------------------------------------------------------------------












def apply_joint_angle(joint:Joint, axis, angle_rad):
    """
    Wendet eine Rotation um eine gegebene Achse auf das Joint-Objekt an.

    :param joint: Joint-Instanz
    :param axis: 3D-Achse als Liste [x, y, z]
    :param angle_rad: Winkel in Radiant
    """
    axis = np.array(axis, dtype = float)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotationsachse darf nicht der Nullvektor sein.")
    axis = axis / np.linalg.norm(axis)  # Normalisieren
    r = R.from_rotvec(axis * angle_rad)  # Rotationsvektor → Quaternion
    quat = r.as_quat()  # [x, y, z, w] Reihenfolge!
    joint.get_renderable().quaternion = tuple(quat)

    for m in joint.mimicers:
        apply_joint_angle(m[0], m[0].axis, angle_rad*m[1])



def apply_joint_rotation(joint:Joint, axis, angle_rad):
    """
    Wendet eine Rotation um eine gegebene Achse und Winkel auf das Joint-Objekt an.

    Die Rotation wird relativ zur bestehenden Grundrotation des Joints berechnet.

    :param joint: Das Joint-Objekt (Instanz von Joint), dessen Rotation gesetzt werden soll.
    :param axis: Die Rotationsachse als Liste oder np.array mit 3 Elementen, z. B. [0, 0, 1].
                 Sollte idealerweise normiert sein (wird intern aber auch normalisiert).
    :param angle_rad: Der Rotationswinkel in Radiant.
    """
    axis = np.array(axis, dtype=np.float64)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotationsachse darf nicht der Nullvektor sein.")
    axis = axis / np.linalg.norm(axis)
    base_rot = R.from_euler("ZYX", joint.get_rotation(), degrees=True)
    axis_rot = R.from_rotvec(axis * angle_rad)
    final_rot = axis_rot * base_rot
    q = final_rot.as_quat()
    joint.get_renderable().quaternion = (q[0], q[1], q[2], q[3])

    #mimicers
    for m in joint.mimicers:
        apply_joint_rotation(m[0], m[0].axis, angle_rad*m[1])
        
        


def apply_joint_rotation_animated(joint:Joint, axis, angle_rad, loop=False, duration:float = 2):
    axis = np.array(axis, dtype=np.float64)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotationsachse darf nicht der Nullvektor sein.")
    axis = axis / np.linalg.norm(axis)
    base_rot = R.from_euler("ZYX", joint.get_rotation(), degrees=True)
    axis_rot = R.from_rotvec(axis * angle_rad)
    final_rot = axis_rot * base_rot
        
    q1 = joint.get_renderable().quaternion
    q2 = final_rot.as_quat()
    tracks = [
        QuaternionKeyframeTrack(name='.quaternion', times=[0,duration], values=[q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]]), 
    ]
    clip:AnimationClip = AnimationClip(tracks=tracks, duration=duration)
    mixer:AnimationMixer = AnimationMixer(joint.get_renderable())
    action:AnimationAction = AnimationAction(mixer, clip, joint.get_renderable())
    if loop==False:
        action.loop = 'LoopOnce'
    action.clampWhenFinished = True
    action.play()


    for mimicer in joint.mimicers:
        apply_joint_rotation_animated(mimicer[0], mimicer[0].axis, angle_rad*mimicer[1], loop=loop)
    return action, q2 , joint








