from __future__ import annotations
import numpy as np
import sympy as sp
import pythreejs as three
from ipywidgets import *
from IPython.display import display
from pythreejs import *
import time
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.sparse import csr_matrix
import visualkinematics_v2.manager as manager
import visualkinematics_v2.util as util
from threading import Timer
from numba import njit
from typing import *
import re
import threading


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
    r = R.from_euler('ZYX', rpy[::-1], degrees=degrees)
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



def compute_dh_matrix_symbolic(theta, d, a, alpha):
    """Erzeuge symbolische DH-Transformationsmatrix"""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])




#-----------------------------------------------------------------------------------------------------------------------------------------------




class DHKinematicModel:
    def __init__(self, dh_parameters:dict, base_to_dh:np.ndarray = np.eye(4), dh_to_tool:np.ndarray = np.eye(4)):  # dh_parameters ist dict von dicts mit theta, d, a, alpha
        self.dh_parameters:dict = dh_parameters
        self.symbolic_thetas:dict = {}
        self.joint_angles:dict = {}
        self.base_to_dh:np.ndarray = base_to_dh
        self.dh_to_tool:np.ndarray = dh_to_tool
        for name, _ in dh_parameters.items():
            self.joint_angles[name] = 0.0
            theta_sym = sp.Symbol(name)
            self.symbolic_thetas[name] = theta_sym
            #print(f"hier: {name}")
        

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
    

    def compute_transforms_symbolic(self):
        """Erzeuge symbolische Transformationsmatrizen für jeden Joint"""
        T_dict = {}
        T = sp.eye(4)
        

        for name, param in self.dh_parameters.items():
            theta_total = self.symbolic_thetas[name] + param["theta"]
            d = param["d"]
            a = param["a"]
            alpha = param["alpha"]

            T_i = self.compute_dh_matrix_symbolic(theta_total, d, a, alpha)
            T = T * T_i
            T_dict[name] = T

        return T_dict
    

    def update_joint_angle(self, name:str, angle_rad:float) -> None:
        self.joint_angles[name] = angle_rad


    def compute_dh_matrix(self, theta:float, d:float, a:float, alpha:float) -> np.ndarray:
        return compute_dh_matrix(theta, d, a, alpha)
    
    def compute_dh_matrix_symbolic(self, theta, d, a, alpha):
        return compute_dh_matrix_symbolic(theta, d, a, alpha)


    def compute_dh_to_tool(self, global_transform_tool:np.ndarray):
        dh_transforms = self.compute_transforms()
        last_key = next(reversed(dh_transforms))
        last_dh = self.base_to_dh @ dh_transforms[last_key]
        return np.linalg.inv(last_dh) @ global_transform_tool


    def inverse_kinematics(self, target_position: np.ndarray, q0: np.ndarray, max_iters=100, tol=1e-4):
        # 1. Symbolisch aufbauen
        dh_transforms = self.compute_transforms_symbolic()
        last_key = next(reversed(dh_transforms))
        last_dh = dh_transforms[last_key]

        base_to_dh_sym = sp.Matrix(self.base_to_dh)
        dh_to_tool_sym = sp.Matrix(self.dh_to_tool)
        full_transform = base_to_dh_sym * last_dh * dh_to_tool_sym

        position = full_transform[:3, 3]

        # 2. Liste der symbolischen Variablen
        q_syms = list(self.symbolic_thetas.values())

        # 3. Symbolische Jacobi-Matrix
        J = sp.Matrix.hstack(*[position.diff(qi) for qi in q_syms])

        # 4. Lambdify: numerische Funktionen erzeugen
        fk_func = sp.lambdify(q_syms, position, 'numpy')
        jacobian_func = sp.lambdify(q_syms, J, 'numpy')

        # 5. Iterativer IK-Algorithmus (Newton-Raphson)
        q = np.array(q0, dtype=float)
        for i in range(max_iters):
            current_pos = np.array(fk_func(*q), dtype=float).flatten()
            error = target_position - current_pos
            #print(f"Iter {i}: Error Norm = {np.linalg.norm(error):.5f}")

            if np.linalg.norm(error) < tol:
                return q  # Lösung gefunden

            J_num = np.array(jacobian_func(*q), dtype=float)
            dq = np.linalg.pinv(J_num, rcond=1e-3) @ error
            q += dq

        raise ValueError("Inverse Kinematik konvergiert nicht")
    

    
    

    def inverse_kinematics6D(self, target_position, target_rotation, q0, max_iters=100, tol=1e-4):
        # Symbolischer Aufbau
        dh_transforms = self.compute_transforms_symbolic()
        last_key = next(reversed(dh_transforms))
        last_dh = dh_transforms[last_key]

        base_to_dh_sym = sp.Matrix(self.base_to_dh)
        dh_to_tool_sym = sp.Matrix(self.dh_to_tool)
        full_transform = base_to_dh_sym * last_dh * dh_to_tool_sym

        pos_expr = full_transform[:3, 3]
        rot_expr = full_transform[:3, :3]

        q_syms = list(self.symbolic_thetas.values())

        # Symbolische Jacobi-Teilmatrix (nur Position)
        J_pos = sp.Matrix.hstack(*[pos_expr.diff(qi) for qi in q_syms])
        fk_pos_func = sp.lambdify(q_syms, pos_expr, 'numpy')
        fk_rot_func = sp.lambdify(q_syms, rot_expr, 'numpy')
        jacobian_func = sp.lambdify(q_syms, J_pos, 'numpy')

        q = np.array(q0, dtype=float)
        for i in range(max_iters):
            pos = np.array(fk_pos_func(*q), dtype=float).flatten()
            rot_mat = np.array(fk_rot_func(*q), dtype=float)

            pos_error = target_position - pos

            # Orientierungsfehler: als Rotationsvektor
            R_current = R.from_matrix(rot_mat)
            R_target = R.from_matrix(target_rotation)
            R_error = R_target * R_current.inv()
            rot_error = R_error.as_rotvec()  # shape: (3,)

            # Gesamtfehlervektor
            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(error) < tol:
                return q  # Lösung gefunden

            # Numerischer Jacobian erweitern: Positionsteil + Orientierungsteil
            J_pos_num = np.array(jacobian_func(*q), dtype=float)

            # Approximierter Rotations-Jacobian: numerisch per finite differences
            delta = 1e-6
            J_rot_num = np.zeros((3, len(q)))
            for j in range(len(q)):
                dq = np.zeros_like(q)
                dq[j] = delta

                q_plus = q + dq
                R_plus = R.from_matrix(np.array(fk_rot_func(*q_plus), dtype=float))
                R_diff = R_plus * R_current.inv()
                rot_vec = R_diff.as_rotvec()
                J_rot_num[:, j] = rot_vec / delta

            J_full = np.vstack((J_pos_num, J_rot_num))

            dq = np.linalg.pinv(J_full, rcond=1e-3) @ error
            q += dq

        raise ValueError("Inverse Kinematik (6D) konvergiert nicht")



    def inverse_kinematics6D_with_limits(self, target_position, target_rotation, q0, max_iters=100, tol=1e-4, joint_mins=None, joint_maxs=None):
        print(f"joint mins: {joint_mins}")
        print(f"joint maxs: {joint_maxs}")
        # Symbolischer Aufbau
        dh_transforms = self.compute_transforms_symbolic()
        last_key = next(reversed(dh_transforms))
        last_dh = dh_transforms[last_key]

        base_to_dh_sym = sp.Matrix(self.base_to_dh)
        dh_to_tool_sym = sp.Matrix(self.dh_to_tool)
        full_transform = base_to_dh_sym * last_dh * dh_to_tool_sym

        pos_expr = full_transform[:3, 3]
        rot_expr = full_transform[:3, :3]

        q_syms = list(self.symbolic_thetas.values())

        # Symbolische Jacobi-Teilmatrix (nur Position)
        J_pos = sp.Matrix.hstack(*[pos_expr.diff(qi) for qi in q_syms])
        fk_pos_func = sp.lambdify(q_syms, pos_expr, 'numpy')
        fk_rot_func = sp.lambdify(q_syms, rot_expr, 'numpy')
        jacobian_func = sp.lambdify(q_syms, J_pos, 'numpy')

        q = np.array(q0, dtype=float)
        for i in range(max_iters):
            pos = np.array(fk_pos_func(*q), dtype=float).flatten()
            rot_mat = np.array(fk_rot_func(*q), dtype=float)

            pos_error = target_position - pos

            # Orientierungsfehler: als Rotationsvektor
            R_current = R.from_matrix(rot_mat)
            R_target = R.from_matrix(target_rotation)
            R_error = R_target * R_current.inv()
            rot_error = R_error.as_rotvec()  # shape: (3,)

            # Gesamtfehlervektor
            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(error) < tol:
                return q  # Lösung gefunden

            # Numerischer Jacobian erweitern: Positionsteil + Orientierungsteil
            J_pos_num = np.array(jacobian_func(*q), dtype=float)

            # Approximierter Rotations-Jacobian: numerisch per finite differences
            delta = 1e-6
            J_rot_num = np.zeros((3, len(q)))
            for j in range(len(q)):
                dq = np.zeros_like(q)
                dq[j] = delta

                q_plus = q + dq
                R_plus = R.from_matrix(np.array(fk_rot_func(*q_plus), dtype=float))
                R_diff = R_plus * R_current.inv()
                rot_vec = R_diff.as_rotvec()
                J_rot_num[:, j] = rot_vec / delta

            J_full = np.vstack((J_pos_num, J_rot_num))

            dq = np.linalg.pinv(J_full, rcond=1e-3) @ error
            q += dq

            # Gelenkgrenzen beachten (falls angegeben)
            if joint_mins is not None and joint_maxs is not None:
                q = np.clip(q, joint_mins, joint_maxs)

        raise ValueError("Inverse Kinematik (6D) konvergiert nicht")





    def forward_kinematics(self) -> np.ndarray:
        dh_transforms = self.compute_transforms()
        last_key = next(reversed(dh_transforms))
        last_dh = dh_transforms[last_key]
        return self.base_to_dh @ last_dh @ self.dh_to_tool


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

    def get_rotation(self, degrees=True) -> np.ndarray:   #INKONSISTENZ MIT DEM KOMPLETTEN REST DER API WEGEN DER REIHENFOLGE DES RÜCKGABEVEKTORS
        q = self.renderable.quaternion  # Quaternion: [x, y, z, w]
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        euler_deg = r.as_euler("ZYX", degrees=degrees)
        return np.array(euler_deg.tolist()[::-1])
    

    def get_quaternion(self):
        return list(self.get_renderable().quaternion)
    
    def get_rotvec(self):
        return R.from_quat(self.get_quaternion()).as_rotvec()
        











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

            joint_lower_limit = None
            joint_upper_limit = None
            if "limit" in joint_element:
                if "lower" in joint_element["limit"]:
                    joint_lower_limit = float(joint_element["limit"]["lower"])
                if "upper" in joint_element["limit"]:
                    joint_upper_limit = float(joint_element["limit"]["upper"])

            if joint_axis is not None:
                joint_axis = joint_axis["xyz"].split()
                joint_axis = [float(x) for x in joint_axis]
            joint : Joint = Joint(joint_element["name"], np.array(joint_axis), np.array(pos), np.array(angles), joint_lower_limit, joint_upper_limit)
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
    def __init__(self, name:str, axis:np.ndarray, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0]), lower_limit:float|None = None, upper_limit:float|None = None):
        super().__init__(name)
        self.is_mimicer:Bool = False
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.renderable : Object3D = three.Group()
        self.axis:np.ndarray = axis
        self.mimicers:list[tuple[Kinematic_Chain_Element, float]] = []
        self.renderable.add(self.frame)
        self.set_position(position)
        self.set_rotation(rotation)
        self.dh_frame = util.create_axes(0.3, show_labels=False, arrow_size=0.1, transparent_arrows=False)
        self.dh_frame.visible = False
        #self.renderable.add(self.dh_frame)

        self.dh_alignment:np.ndarray = np.eye(4)  # später ggf. mit echten Werten setzen

    def add_mimicer(self, mimicer_and_multiplier:tuple[Kinematic_Chain_Element, float]):
        if not isinstance(mimicer_and_multiplier[0], Joint) : raise RuntimeError(f"jetzt ist klar, dass nicht nur Joints Mimicer sein können, sondern auch {type(mimicer_and_multiplier[0])}")
        mimicer_and_multiplier[0].is_mimicer = True
        self.mimicers.append(mimicer_and_multiplier)
    

    def get_previous_joint_in_chain(self) -> Joint|None:
        current = self.parent
        while current is not None:
            if isinstance(current, Joint):
                return current
            current = current.parent
        return current



    def _create_theta_slider(self, num, value, callback: Callable[[], None] = None):
        if self.axis is None : return None
        sign = 1
        if sum(self.axis) < 0 : sign *= -1
        z = self.dh_alignment[:3, -2]     #vorletzte spalte, erste 3 elemente (Spaltenvektor z-achse)
        if sum(z) < 0 : sign *= -1

        
        renderable = self
        if hasattr(self, "get_renderable"):
            renderable = self.get_renderable()

        layout1 = widgets.Layout(
                border='1px solid gray',
                padding='2px',
                height='40px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
            )  
        min = -180
        max = 180
        if self.lower_limit is not None:
            min = np.rad2deg(self.lower_limit) * sign
        if self.upper_limit is not None:
            max = np.rad2deg(self.upper_limit) * sign
        if min > max:
            tmp = max
            max = min
            min = tmp
        theta_rot_slider = FloatSlider(min=min, max=max, step=0.1, description=f'Theta {num}', layout=layout1)
        rot = R.from_quat(list(renderable.quaternion))
        euler = rot.as_euler("XYZ", degrees=True) 
        
        if abs(self.axis[0]) == 1: theta_rot_slider.value = euler[0]
        elif abs(self.axis[1]) == 1: theta_rot_slider.value = euler[1]
        elif abs(self.axis[2]) == 1: theta_rot_slider.value = euler[2]
        
    
        rot = R.from_quat(list(renderable.quaternion))
        euler = rot.as_euler("XYZ", degrees=True)
        
        def _on_rot_slider(change):
            if abs(self.axis[0]) == 1:
                util.set_rotation(renderable, [theta_rot_slider.value * sign, euler[1], euler[2]], "XYZ")
            elif abs(self.axis[1]) == 1:
                util.set_rotation(renderable, [euler[0], theta_rot_slider.value * sign, euler[2]], "XYZ")
            elif abs(self.axis[2]) == 1:
                util.set_rotation(renderable, [euler[0], euler[1], theta_rot_slider.value * sign], "XYZ")
            if callback is not None : callback()
                  
        theta_rot_slider.observe(_on_rot_slider, names="value")
        return [self.name, theta_rot_slider]


class Link(Kinematic_Chain_Element):
    def __init__(self, name:str, mesh:Mesh, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0])) -> None:
        super().__init__(name)
        self.renderable:Mesh = mesh
        self.renderable.add(self.frame)
        self.set_position(position)
        self.set_rotation(rotation)
        
    

        










#---------------------------------------------MANIPULATOR-------------------------------------------------------------------------









class Manipulator:
    def __init__(self, name:str, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        self.k0 = None
        self.inspector:Manipulator.Inspector|None = None
        self.environment:util.Environment|None = None
        self.name:str = name
        self.links:list[Link] = []
        self.joints:list[Joint] = []
        self.dh:DHKinematicModel|None = None
        xacro_filepath:str = manager.find_xacro_filepath_by_robot_name(name)
        urdf:str = manager.xacro_to_urdf_string(xacro_filepath)
        self.urdf_dictionary:dict = manager.parse_urdf(urdf)
        self.tool:Tool|None = None
        self.learned_poses = self._load_learned_poses()
        self.tcp_target = None
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
        

        self.base_to_baselink_joint:Joint = self.base.parent
        self.baselink_to_base_joint:Joint = self.base.parent
        if tool_name!="":
            t = self.get_link_by_name("tool0")
            t.add(self.tool)
        

        self.link_to_tool_joint:Joint = self.find_link_to_tool_joint()

    def set_environment(self, env:util.Environment):
        self.environment = env



    def add_inspector(self, env):
        self.inspector = self.Inspector(env, self)
        env.add_widget(self.inspector.widget)
        return self.inspector









#--------------------------------------------INSPECTOR--------------------------------------------------------------------------------







    class Inspector:
        layout = widgets.Layout(
            #border='1px solid gray',
            padding='2px',
            height='40px',
            overflow='hidden',  # Scrollen deaktivieren
            flex='none'
            )

        def __init__(self, env:util.Environment, manipulator:Manipulator):
            self.gizmo_controls:util.Environment.Gizmo_Controls = None
            self.sliders = {}
            self.manipulator = manipulator
            self.content = []
            self.pose_names = sorted(set(obj["name"] for obj in manipulator.learned_poses))
            self.pose_dropdown:widgets.Dropdown = widgets.Dropdown(
                options=self.pose_names,
                style={'description_width': 'initial'})

            self.save_pose_textfield:widgets.Text = widgets.Text(
                value='',
                placeholder='Name für Pose',
                #description='Eingabe:',
                disabled=False
            )
            self.save_button:widgets.Button = widgets.Button(
                description='Pose Speichern',
                disabled=False,
                button_style='',
                tooltip='Speichert Die Pose, sodass sie in Zukunft wieder eingenommen werden kann.',
                icon='save'
            )

            self.save_button.on_click(self._on_click_save_button)

            self.take_position_button:widgets.Button = widgets.Button(
                description='Pose Einnehmen',
                disabled=False,
                button_style='',
                tooltip='Überführt den Roboter in die Ausgewählte Pose!',
                icon='play'
            )

        
            self.take_position_button.on_click(self._on_click_take_position_button)

            hbox2 = widgets.HBox([self.pose_dropdown, self.take_position_button], layout=self.__class__.layout)

            # Horizontal anordnen
            hbox1 = widgets.HBox([self.save_pose_textfield, self.save_button], layout=self.__class__.layout)
            self.content.append(hbox1)
            self.content.append(hbox2)
            
            self._create_theta_sliders()

            manipulator.tcp_target = util.create_axes(0.3, 0.1, False, "", 0.2)
            if manipulator.dh is not None:
                util.apply_transformation_matrix(manipulator.tcp_target, manipulator.get_global_tcp_transform())
            env.add(manipulator.tcp_target)
            
            self.gizmo_controls = env.Gizmo_Controls(manipulator.tcp_target, True, True, False, "TCP-Target", 3, 3, 3, -3, -3, -3, widgets_vertical=True, continuous_update=True, callback=self._on_gizmo_controls)
            self.content.append(self.gizmo_controls.widget)
            self.inverse_kinematic_button:widgets.Button = widgets.Button(
                description = "Pose Suchen",
                tooltip='Berechnet die inverse Kinematik und überführt den Roboter in die gefundene Pose!',
                layout = widgets.Layout(
                        #border='1px solid gray',
                        padding='2px',
                        height='40px',
                        width='40',
                        overflow='hidden',  # Scrollen deaktivieren
                        flex='none'
                        )
                )
            self.inverse_kinematic_button.on_click(self._on_inverse_kinematic_button)
            self.content.append(self.inverse_kinematic_button)
            self.widget = widgets.VBox(children = self.content)
        


        def _create_theta_sliders(self):
            num = 1
            for j in self.manipulator.joints:
                if j.is_mimicer: continue
                name_and_slider = j._create_theta_slider(num, value=0, callback=self._on_theta_slider)
                if name_and_slider != None: 
                    self.content.append(name_and_slider[1])
                    self.sliders[name_and_slider[0]] = name_and_slider[1]
                num += 1


        def _on_inverse_kinematic_button(self, button):
            r = R.from_quat(list(self.manipulator.tcp_target.quaternion)).as_matrix()
            p = self.manipulator.tcp_target.position
            q0 = np.array(list(self.manipulator.dh.joint_angles.values()))
            q_sol = self.manipulator.dh.inverse_kinematics6D_with_limits(p, r, q0, 10, 0.1)
            self.manipulator.animate_by_theta(q_sol, 0.75, 1, True, False)
            self.manipulator.update_dh_angles()
            #------
            for name, slider in self.sliders.items():
                if name in self.manipulator.dh.joint_angles:
                    angle= self.manipulator.dh.joint_angles[name]
                    #display(name)
                    #display((angle / (2*np.pi) * 360))
                    #slider.value = angle / (2*np.pi) * 360


        def _on_gizmo_controls(self):
            pass
            
                    




        def _on_click_take_position_button(self, button:widgets.Button):
            try:
                button.description = "Pose Einnehmen"
                button.icon='pause'
                self.manipulator.animate_by_learned_pose(name = self.pose_dropdown.value, synchronous=True, duation=4)
                
            except Exception as e:
                info = self.manipulator.environment.add_info(f"Beim Einnehmen der Pose ist ein Fehler aufgetreten!: {e}")
            button.description = "Pose Einnehmen"
            button.icon='play'



        def _on_theta_slider(self):
                transform = self.manipulator.get_global_tcp_transform()
                util.apply_transformation_matrix(self.manipulator.tcp_target, transform)
                pos = transform[:3, 3]
                self.gizmo_controls.set_translation_silently(pos)
                if self.gizmo_controls.local_space_check_box.value:
                    euler = util.quaternion_to_euler(self.manipulator.tcp_target.quaternion[0], self.manipulator.tcp_target.quaternion[1], self.manipulator.tcp_target.quaternion[2], self.manipulator.tcp_target.quaternion[3], self.gizmo_controls.rotation_order_dropdown.value)
                    euler = util.order_angles(euler, self.gizmo_controls.rotation_order_dropdown.value, "XYZ")
                    self.gizmo_controls.set_rotation_silently(euler)
                else :
                    euler = util.quaternion_to_euler(self.manipulator.tcp_target.quaternion[0], self.manipulator.tcp_target.quaternion[1], self.manipulator.tcp_target.quaternion[2], self.manipulator.tcp_target.quaternion[3], self.gizmo_controls.rotation_order_dropdown.value[::-1])
                    euler = util.order_angles(euler, self.gizmo_controls.rotation_order_dropdown.value[::-1], "XYZ")
                    self.gizmo_controls.set_rotation_silently(euler)




        def _on_click_save_button(self, button:widgets.Button):
            info = None
            try:
                self.manipulator.learn(pose_name = self.save_pose_textfield.value)
                opts = list(self.pose_dropdown.options)
                opts.append(self.save_pose_textfield.value)
                self.pose_dropdown.options = opts
                button.description = "Gespeichert!"
                button.icon='check'
            
            except Exception as e:
                info = self.manipulator.environments.add_info(f"Beim Speichern der Pose ist ein Fehler aufgetreten!: {e}")
            def reset():
                time.sleep(1.5)
                button.description = "Pose Speichern"
                button.icon='save'
            threading.Thread(target=reset).start() 




#--------------------------------------------------------------------------------------------------------





    def learn(self, pose_name: str, thetas:list|None = None):
        display(f"POSEN: {self.learned_poses}")
        exists = any(obj["name"] == pose_name for obj in self.learned_poses)
        if exists : raise RuntimeError("Pose mit diesem Namen existiert bereits")
        filepath = f"{manager.learn_path}/{self.name}.json"
        q = []
        if thetas is None:
            self.update_dh_angles()
            for key, value in self.dh.joint_angles.items():
                q.append(value)
        else : q = thetas
        # Neue Pose
        new_pose = {
            "name": pose_name,
            "theta": list(q)  # z. B. [0.1, -1.2, ...]
        }

        # Ordner erstellen, falls nicht vorhanden
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Bestehende Datei einlesen, falls vorhanden
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            data = []

        # Neue Pose anhängen
        data.append(new_pose)

        # Zurückschreiben
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        self.learned_poses = self._load_learned_poses()
        



    def _load_learned_poses(self):
        filepath = f"{manager.learn_path}/{self.name}.json"
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Lern-Datei '{filepath}' ist beschädigt oder leer.")
                return []
        else:
            return []


    def get_learned_pose(self, name: str):
        for pose in self.learned_poses:
            if pose["name"] == name:
                return pose["theta"]
        raise ValueError(f"Pose '{name}' nicht gefunden.")

    def update_dh_angles(self):
        for name, _ in self.dh.joint_angles.items():
            current_joint = self.get_joint_by_name(name)
            prev_joint = current_joint.get_previous_joint_in_chain()
            
            new_angle = 0.0
            sign = 1
            z = prev_joint.dh_alignment[:3, -2]     #vorletzte spalte, erste 3 elemente (Spaltenvektor z-achse)
            if prev_joint is None or prev_joint.axis is None:
                    new_angle = 0.0
            else:
                rotvec = prev_joint.get_rotvec()
                new_angle = np.linalg.norm(rotvec)
                if sum(rotvec) < 0: sign *=-1
                if sum(z) < 0: sign *= -1
            self.dh.update_joint_angle(name, new_angle * sign)
            #print(f"joint:{name:<15}\t\tprev_joint:{prev_joint.name:<15}\t\trad:{new_angle}\t\tdeg:{new_angle*(180/np.pi)}\t\trotvec:{rotvec}\t\trotation:{prev_joint.get_rotation(False)}\t\tprev_axis:{prev_joint.axis}\t\taxis:{current_joint.axis}")
            #print(f"DH_align:\n{prev_joint.dh_alignment}")


    def get_global_tcp_transform(self) -> np.ndarray:
        self.update_dh_angles()
        return self.dh.forward_kinematics()


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


    def animate_by_learned_pose(self, name:str, duation:float = 1.0, quality:float = 1.0, synchronous:bool = False, with_tcp_target:bool = True):
        pose = self.get_learned_pose(name)
        self.animate_by_theta(pose, duation, quality, synchronous, with_tcp_target)
        

    def animate_by_theta(self, theta:list, duration:float = 1.0, quality:float = 1.0, synchronous:bool = False, with_tcp_target:bool = True):
        joints:list = []
        angles_rad:list = []
        i = 0
        for name, _ in self.dh.joint_angles.items():
            current_joint = self.get_joint_by_name(name)
            prev_joint = current_joint.get_previous_joint_in_chain()
            sign = 1
            z = prev_joint.dh_alignment[:3, -2]     #vorletzte spalte, erste 3 elemente (Spaltenvektor z-achse)
            
            if sum(z) < 0: sign *= -1
            if sum(prev_joint.axis) < 0: sign *= -1
            joints.append(prev_joint)
            angles_rad.append(theta[i]*sign)
            i+=1
        self.animate_experimental(joints, angles_rad, duration, quality, False, synchronous, with_tcp_target)


    def animate_experimental(self, joints:list, angles_rad:list, duration:float = 1.0, quality:float = 1, add_on_baserotation:bool = True, synchronous:bool = False, with_tcp_target:bool = True):
        def _animate_experimental_task(joints:list, angles_rad:list, duration:float = 1.0, quality:float = 1):
            def add_mimicers(current:Joint, current_angle_rad, joints:list):
                joints.append(current)
                angles_rad.append(current_angle_rad)
                for m in current.mimicers:
                    add_mimicers(current=m[0], current_angle_rad=m[1] * current_angle_rad, joints=joints)

            if with_tcp_target:
                util.set_rotation(self.tcp_target, [0,0,0], "XYZ")
                util.set_translation(self.tcp_target, [0,0,0])
                self.tool.renderable.add(self.tcp_target)

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
                base_rot = R.from_quat(joint.get_quaternion())
                #base_rot = R.from_euler("ZYX", joint.get_rotation(), degrees=True)
                axis_rot = R.from_rotvec(axis * angle_rad)
                final_rot = None
                if add_on_baserotation : final_rot = axis_rot * base_rot
                else : final_rot = axis_rot
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

            if self.environment is not None and with_tcp_target:
                util.apply_transformation_matrix(self.tcp_target, self.get_global_tcp_transform())
                self.environment.add(self.tcp_target) 


        thread = threading.Thread(target=_animate_experimental_task, args=(joints, angles_rad, duration, quality))
        thread.start()
        if synchronous : thread.join()  
     



    




    def apply_DH_model(self, dh:DHKinematicModel) -> None:
        self.dh = dh
        DH_transforms:dict = dh.compute_transforms()
        global_transforms:dict = self.compute_global_transform()
        for name, transform in DH_transforms.items():
            joint:Joint = self.get_joint_by_name(name)
            joint.dh_alignment = np.linalg.inv(global_transforms[name]) @ (dh.base_to_dh @ transform)

            joint.dh_frame.position = joint.get_position()
            joint.parent.renderable.add(joint.dh_frame)

        self.k0 = util.create_axes(0.3, show_labels=False, arrow_size=0.1, transparent_arrows=False)     #nicht schön...das muss in zukunft anders gelöst werden! der manipulator sollte keine variable k0 haben
        self.k0.visible = False
        util.apply_transformation_matrix(self.k0, dh.base_to_dh)
        self.base_link.renderable.add(self.k0)
        if self.tcp_target is not None:
            util.apply_transformation_matrix(self.tcp_target, self.get_global_tcp_transform())




        




    

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
            joint_lower_limit = None
            joint_upper_limit = None
            if "limit" in joint_element:
                if "lower" in joint_element["limit"]:
                    joint_lower_limit = float(joint_element["limit"]["lower"])
                if "upper" in joint_element["limit"]:
                    joint_upper_limit = float(joint_element["limit"]["upper"])
            if joint_axis is not None:
                joint_axis = joint_axis["xyz"].split()
                joint_axis = [float(x) for x in joint_axis]

            if joint_axis is not None:
                joint_axis = np.array(joint_axis)
            joint:Joint = Joint(joint_element["name"], joint_axis, pos, angles, joint_lower_limit, joint_upper_limit)
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
        self.k0.visible = show
        current = self.base_link
        if isinstance(current, Joint):
            current.dh_frame.visible=show
            pos = [current.dh_frame.position[0], current.dh_frame.position[1], current.dh_frame.position[2]]
            util.apply_transformation_matrix(current.dh_frame, current.dh_alignment)
            util.translate(current.dh_frame, pos)
        while(len(current.children) > 0):
            current = current.children[0]
            if isinstance(current, Joint):
                current.dh_frame.visible=show
                pos = [current.dh_frame.position[0], current.dh_frame.position[1], current.dh_frame.position[2]]
                util.apply_transformation_matrix(current.dh_frame, current.dh_alignment)
                util.translate(current.dh_frame, pos)







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

    

    def find_link_to_tool_joint(self):
        return self.get_link_by_name("tool0").parent
        
        


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
            self.compute_global_transform(child, current_transform, global_transforms, with_print)
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
    base_rot = R.from_euler("ZYX", joint.get_rotation()[::-1], degrees=True)
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
    base_rot = R.from_euler("ZYX", joint.get_rotation()[::-1], degrees=True)
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








