from __future__ import annotations
import numpy as np
import sympy as sp
import pythreejs as three
from ipywidgets import *
from IPython.display import display
from pythreejs import *
import time
from scipy.spatial.transform import Rotation as R, Slerp
from visualkinematics_v2 import manager
from visualkinematics_v2 import util
from threading import Timer
from numba import njit
from typing import *
import re
import threading
import numba
import ipyevents

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Gimbal lock detected.*")

#fps_text = widgets.Text("empty")

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
    '''
    Wandelt eine Pose, bestehend aus Translation (xyz) und Euler-Winkeln (rpy), 
    in eine homogene 4x4-Transformationsmatrix um.

    :param xyz: Translationsvektor als NumPy-Array mit [x, y, z].
    :param rpy: Rotationswinkel als NumPy-Array [roll, pitch, yaw].
    :param degrees: Boolescher Wert, der angibt, ob die Winkel in Grad (True) 
                    oder Radiant (False) angegeben sind.

    :return: Eine 4x4-Transformationsmatrix (NumPy-Array), die Translation und 
             Rotation der Pose beschreibt.
    '''
    r = R.from_euler('ZYX', rpy[::-1], degrees=degrees)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = xyz
    return T


@njit
def compute_dh_matrix(theta:float, d:float, a:float, alpha:float):
    '''
    Berechnet die homogene 4x4-Transformationsmatrix anhand der 
    Denavit-Hartenberg-Parameter.

    :param theta: Rotationswinkel um die z-Achse (in Radiant).
    :param d: Verschiebung entlang der z-Achse.
    :param a: Verschiebung entlang der x-Achse (Länge des Gelenkarms).
    :param alpha: Rotationswinkel um die x-Achse (in Radiant).

    :return: Eine 4x4-Transformationsmatrix (NumPy-Array) gemäß der
             Denavit-Hartenberg-Konvention.
    '''
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,       sa,      ca,     d],
        [0.0,        0.0,       0.0,     1.0]
    ], dtype=np.float64)



def compute_dh_matrix_symbolic(theta, d, a, alpha):
    '''
    Erzeugt eine symbolische 4x4-Transformationsmatrix anhand der 
    Denavit-Hartenberg-Parameter.

    :param theta: Rotationswinkel um die z-Achse (SymPy-Symbol oder -Ausdruck).
    :param d: Verschiebung entlang der z-Achse (SymPy-Symbol oder -Ausdruck).
    :param a: Verschiebung entlang der x-Achse (SymPy-Symbol oder -Ausdruck).
    :param alpha: Rotationswinkel um die x-Achse (SymPy-Symbol oder -Ausdruck).

    :return: Eine 4x4-Transformationsmatrix als SymPy-Matrix, 
             gemäß der Denavit-Hartenberg-Konvention.
    '''
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])




#-----------------------------------------------------------------------------------------------------------------------------------------------




class DHKinematicModel:
    """
    Repräsentiert ein kinematisches Modell basierend auf Denavit-Hartenberg (DH)-Parametern.

    Diese Klasse ermöglicht die Vorwärts- und Inverse-Kinematik für einen seriellen Manipulator 
    unter Verwendung von DH-Parametern. Sowohl numerische als auch symbolische Berechnungen werden unterstützt.
    
    Funktionen:
    - Berechnung der Vorwärtskinematik (Forward Kinematics) für gegebene Gelenkwinkel.
    - Berechnung der Inversen Kinematik (Inverse Kinematics) für gewünschte Zielpositionen
      und -orientierungen, einschließlich Unterstützung für Gelenkgrenzen.
    - Erstellung symbolischer DH-Transformationen und Jacobians.
    - Unterstützung für 6D-Inverse-Kinematik unter Berücksichtigung von Position und Orientierung.
    - Funktionen zur Berechnung von Transformationsmatrizen zwischen Basis, DH-Frames und Werkzeug.
    
    Attribute:
    - dh_parameters (dict): DH-Parameter für jedes Gelenk als Dictionary mit Keys 'theta', 'd', 'a', 'alpha'.
    - symbolic_thetas (dict): Symbolische Variablen für die Gelenkwinkel (sympy.Symbols).
    - joint_angles (dict): Aktuelle Gelenkwinkel in Radiant.
    - base_to_dh (np.ndarray): 4x4-Transformationsmatrix von Basis zu erstem DH-Gelenk.
    - dh_to_tool (np.ndarray): 4x4-Transformationsmatrix vom letzten DH-Gelenk zum Werkzeug.
    - fk (Callable): Numerische Funktion der vollständigen Vorwärtskinematik.
    - fk_pos_func (Callable): Numerische Funktion für die Position des Endeffektors.
    - fk_rot_func (Callable): Numerische Funktion für die Orientierung des Endeffektors.
    - jacobian_func (Callable): Numerische Funktion der Jacobi-Matrix der Position.

    Hinweis:
    - Die Klasse nutzt sympy für symbolische Berechnungen und numpy für numerische Auswertungen.
    - Die Inverse-Kinematik basiert auf einem iterativen Newton-Raphson-Verfahren.
    - Unterstützung für 6D-Kinematik beinhaltet die Behandlung von Orientierungen über Rotationsvektoren.
    """

    def __init__(self, dh_parameters:dict, base_to_dh:np.ndarray = np.eye(4), dh_to_tool:np.ndarray = np.eye(4)):  # dh_parameters ist dict von dicts mit theta, d, a, alpha
        """
        Initialisiert ein DH-Kinematikmodell mit gegebenen DH-Parametern und optionalen
        Transformationsmatrizen für Basis und Werkzeug.

        Erstellt interne Dictionaries für symbolische Gelenkwinkel und aktuelle Gelenkwinkel
        und initialisiert Platzhalter für Vorwärtskinematik- und Jacobian-Funktionen.

        :param dh_parameters: Dictionary von Dictionaries mit den DH-Parametern 'theta', 'd', 'a', 'alpha' für jedes Gelenk.
        :param base_to_dh: 4x4 Transformationsmatrix vom Basisrahmen zum ersten DH-Rahmen (Standard: Einheitsmatrix).
        :param dh_to_tool: 4x4 Transformationsmatrix vom letzten DH-Rahmen zum Werkzeug (Standard: Einheitsmatrix).
        """
        self.dh_parameters:dict = dh_parameters
        self.symbolic_thetas:dict = {}
        self.joint_angles:dict = {}
        self.base_to_dh:np.ndarray = base_to_dh
        self.dh_to_tool:np.ndarray = dh_to_tool
        self.fk = None
        self.fk_pos_func = None
        self.fk_rot_func = None
        self.jacobian_func = None
        for name, _ in dh_parameters.items():
            self.joint_angles[name] = 0.0
            theta_sym = sp.Symbol(name)
            self.symbolic_thetas[name] = theta_sym
            #print(f"hier: {name}")
    

    @staticmethod
    def compute_base_to_dh(base_transform, k0):
        '''
        Berechnet die Transformationsmatrix vom Roboterbasis-Koordinatensystem zum ersten DH-Gelenk.

        :param base_transform: 4x4 Transformationsmatrix der Roboterbasis.
        :param k0: 4x4 Transformationsmatrix des ersten DH-Gelenks im Weltkoordinatensystem.
        :return: 4x4 Transformationsmatrix vom Basis-Koordinatensystem zum ersten DH-Gelenk.
        '''
        base_transform_inv = np.linalg.inv(base_transform)
        return base_transform_inv @ k0



    def compute_transforms(self) -> dict: 
        '''
        Berechnet die numerischen Transformationsmatrizen für jedes Gelenk basierend auf den aktuellen Gelenkwinkeln und DH-Parametern.

        Für jedes Gelenk wird die 4x4 Transformationsmatrix von der Basis bis zum jeweiligen Gelenk numerisch berechnet
        und in einem Dictionary gespeichert.

        :return: Dictionary, das jedem Gelenknamen die entsprechende 4x4 numerische Transformationsmatrix zuordnet.
        '''
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
        '''
        Erzeugt symbolische Transformationsmatrizen für jedes Gelenk basierend auf den DH-Parametern.

        Für jedes Gelenk wird die Transformationsmatrix von der Basis bis zum jeweiligen Gelenk
        symbolisch berechnet und in einem Dictionary gespeichert.

        :return: Dictionary, das jedem Gelenknamen die entsprechende 4x4 symbolische Transformationsmatrix zuordnet.
        '''
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
        """
        Aktualisiert den Winkel eines bestimmten Gelenks.

        :param name: Name des Gelenks, dessen Winkel geändert werden soll.
        :param angle_rad: Neuer Winkel in Radiant.
        """
        self.joint_angles[name] = angle_rad




    def compute_dh_matrix(self, theta:float, d:float, a:float, alpha:float) -> np.ndarray:
        """
        Berechnet die Denavit-Hartenberg-Transformationsmatrix für ein Gelenk.

        :param theta: Gelenkwinkel um die Z-Achse in Radiant.
        :param d: Verschiebung entlang der Z-Achse.
        :param a: Verschiebung entlang der X-Achse.
        :param alpha: Verdrehung um die X-Achse in Radiant.
        :return: 4x4-DH-Transformationsmatrix als NumPy-Array.
        """
        return compute_dh_matrix(theta, d, a, alpha)
    




    def compute_dh_matrix_symbolic(self, theta, d, a, alpha):
        """
        Erzeugt die Denavit-Hartenberg-Transformationsmatrix symbolisch.

        :param theta: Gelenkwinkel um die Z-Achse (symbolisch oder numerisch).
        :param d: Verschiebung entlang der Z-Achse (symbolisch oder numerisch).
        :param a: Verschiebung entlang der X-Achse (symbolisch oder numerisch).
        :param alpha: Verdrehung um die X-Achse (symbolisch oder numerisch).
        :return: 4x4-Symbolmatrix (sympy.Matrix) der DH-Transformation.
        """
        return compute_dh_matrix_symbolic(theta, d, a, alpha)






    def compute_dh_to_tool(self, global_transform_tool:np.ndarray):
        """
        Berechnet die Transformationsmatrix vom letzten DH-Frame zum Werkzeug (Tool) in globalen Koordinaten.

        :param global_transform_tool: 4x4 Transformationsmatrix des Werkzeugs in globalen Koordinaten.
        :return: 4x4 Transformationsmatrix vom letzten DH-Frame zum Werkzeug.
        """
        dh_transforms = self.compute_transforms()
        last_key = next(reversed(dh_transforms))
        last_dh = self.base_to_dh @ dh_transforms[last_key]
        return np.linalg.inv(last_dh) @ global_transform_tool









    def inverse_kinematics(self, target_position: np.ndarray, q0: np.ndarray, max_iters=100, tol=1e-4):
        '''
        Berechnet die Gelenkwinkel (q) für eine gewünschte Zielposition des Endeffektors 
        mittels iterativer Inverser Kinematik (Newton-Raphson-Verfahren).

        :param target_position: Zielposition des Endeffektors als NumPy-Array [x, y, z].
        :param q0: Startwert für die Gelenkwinkel als NumPy-Array.
        :param max_iters: Maximale Anzahl an Iterationen (Standard: 100).
        :param tol: Abbruchschwelle für den Positionsfehler (Standard: 1e-4).

        :return: Ein NumPy-Array mit den berechneten Gelenkwinkeln (q), das die Zielposition 
                erreicht (innerhalb der Toleranz).
        :raises ValueError: Falls die Inverse Kinematik nicht innerhalb der 
                            maximalen Iterationen konvergiert.
        '''
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
    

    
    





    def inverse_kinematics6D(self, target_position, target_rotation, q0:np.ndarray, max_iters=100, tol=1e-4):
        '''
        Berechnet die Gelenkwinkel (q) für eine gewünschte Zielposition und Zielorientierung 
        des Endeffektors mittels iterativer 6D-Inverser Kinematik (Newton-Raphson-Verfahren).

        :param target_position: Zielposition des Endeffektors als NumPy-Array [x, y, z].
        :param target_rotation: Zielorientierung des Endeffektors als 3x3-Rotationsmatrix.
        :param q0: Startwert für die Gelenkwinkel als NumPy-Array.
        :param max_iters: Maximale Anzahl an Iterationen (Standard: 100).
        :param tol: Abbruchschwelle für den Gesamtfehler (Position + Orientierung) (Standard: 1e-4).

        :return: Ein NumPy-Array mit den berechneten Gelenkwinkeln (q), das Zielposition 
                und -orientierung innerhalb der Toleranz erreicht.
        :raises ValueError: Falls die 6D-Inverse-Kinematik nicht innerhalb der 
                            maximalen Iterationen konvergiert.

        Hinweis:
        - Die Orientierung wird über Rotationsvektoren (axis-angle) behandelt.
        - Der Jacobian für die Rotation wird numerisch über finite Differenzen approximiert.
        '''
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







    def inverse_kinematics6D_with_limits(self, target_position, target_rotation, q0:np.ndarray, max_iters=100, tol=1e-4, joint_mins=None, joint_maxs=None):
        '''
        Berechnet die Gelenkwinkel (q) für eine gewünschte Zielposition und Zielorientierung 
        des Endeffektors mittels iterativer 6D-Inverser Kinematik (Newton-Raphson-Verfahren) 
        unter Berücksichtigung von Gelenkgrenzen.

        :param target_position: Zielposition des Endeffektors als NumPy-Array [x, y, z].
        :param target_rotation: Zielorientierung des Endeffektors als 3x3-Rotationsmatrix.
        :param q0: Startwert für die Gelenkwinkel als NumPy-Array.
        :param max_iters: Maximale Anzahl an Iterationen (Standard: 100).
        :param tol: Abbruchschwelle für den Gesamtfehler (Position + Orientierung) (Standard: 1e-4).
        :param joint_mins: Optionales NumPy-Array mit minimalen Gelenkwinkeln.
        :param joint_maxs: Optionales NumPy-Array mit maximalen Gelenkwinkeln.

        :return: Ein NumPy-Array mit den berechneten Gelenkwinkeln (q), das Zielposition 
                und -orientierung innerhalb der Toleranz erreicht und die Gelenkgrenzen beachtet.
        :raises ValueError: Falls die 6D-Inverse-Kinematik nicht innerhalb der 
                            maximalen Iterationen konvergiert.

        Hinweis:
        - Die Orientierung wird über Rotationsvektoren (axis-angle) behandelt.
        - Der Jacobian für die Rotation wird numerisch über finite Differenzen approximiert.
        - Gelenkwinkel werden nach jedem Schritt auf die angegebenen Grenzen geklippt, falls gesetzt.
        '''
        
        for _ in range(max_iters):
            pos = np.array(self.fk_pos_func(*q0), dtype=float).flatten()
            rot_mat = np.array(self.fk_rot_func(*q0), dtype=float)

            pos_error = target_position - pos

            # Orientierungsfehler: als Rotationsvektor
            R_current = R.from_matrix(rot_mat)
            R_target = R.from_matrix(target_rotation)
            R_error = R_target * R_current.inv()
            rot_error = R_error.as_rotvec()  # shape: (3,)

            # Gesamtfehlervektor
            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(error) < tol:
                return q0  # Lösung gefunden

            # Numerischer Jacobian erweitern: Positionsteil + Orientierungsteil
            J_pos_num = np.array(self.jacobian_func(*q0), dtype=float)

            # Approximierter Rotations-Jacobian: numerisch per finite differences
            delta = 1e-6
            J_rot_num = np.zeros((3, len(q0)))
            for j in range(len(q0)):
                dq = np.zeros_like(q0)
                dq[j] = delta

                q_plus = q0 + dq
                R_plus = R.from_matrix(np.array(self.fk_rot_func(*q_plus), dtype=float))
                R_diff = R_plus * R_current.inv()
                rot_vec = R_diff.as_rotvec()
                J_rot_num[:, j] = rot_vec / delta

            J_full = np.vstack((J_pos_num, J_rot_num))

            dq = np.linalg.pinv(J_full, rcond=1e-3) @ error
            q0 += dq

            # Gelenkgrenzen beachten (falls angegeben)
            if joint_mins is not None and joint_maxs is not None:
                q0 = np.clip(q0, joint_mins, joint_maxs)

        raise ValueError("Inverse Kinematik (6D) konvergiert nicht")







    
    def forward_kinematics(self) -> np.ndarray:
        '''
        Berechnet die Vorwärtskinematik des Roboters.

        Verwendet die aktuell gesetzten Gelenkwinkel (self.joint_angles), um die
        vollständige 4x4-Transformationsmatrix des Endeffektors zu berechnen.

        :return: 4x4 NumPy-Array der Transformationsmatrix (Rotation + Translation) des Endeffektors.
        '''
        arr = self.fk(*list(self.joint_angles.values()))
        return np.array(arr)
    



    def init_functions(self):
        '''
        Initialisiert die Vorwärtskinematik- und Jacobian-Funktionen des Roboters.

        Erstellt symbolische Ausdrücke für die vollständige Transformationsmatrix, die Position
        und Orientierung des Endeffektors sowie die Jacobi-Matrix für die Position. Anschließend
        werden numerische Funktionen (lambdify) erzeugt, die in den Inverse-Kinematik-Funktionen
        verwendet werden.

        Erstellt:
        - self.fk: Vollständige 4x4-Transformationsmatrix als Funktion der Gelenkwinkel.
        - self.fk_pos_func: 3D-Positionsvektor des Endeffektors als Funktion der Gelenkwinkel.
        - self.fk_rot_func: 3x3-Rotationsmatrix des Endeffektors als Funktion der Gelenkwinkel.
        - self.jacobian_func: Jacobi-Matrix der Position (3xN) als Funktion der Gelenkwinkel.

        Hinweis:
        - Alle Funktionen werden mit NumPy kompatibel erstellt.
        - Die symbolischen Variablen stammen aus self.symbolic_thetas.
        '''
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

        self.fk = sp.lambdify(q_syms, full_transform, modules = "numpy")

        q_syms = list(self.symbolic_thetas.values())

        # Symbolische Jacobi-Teilmatrix (nur Position)
        J_pos = sp.Matrix.hstack(*[pos_expr.diff(qi) for qi in q_syms])
        self.fk_pos_func = sp.lambdify(q_syms, pos_expr, 'numpy')
        self.fk_rot_func = sp.lambdify(q_syms, rot_expr, 'numpy')
        self.jacobian_func = sp.lambdify(q_syms, J_pos, 'numpy')

    


#----------------------------------------------------------------------------------------------------------------







class Kinematic_Chain_Element:
    """
    Repräsentiert ein Element einer kinematischen Kette in 3D, das eine Hierarchie von Kindern verwalten kann 
    und Position, Rotation sowie Darstellung über ein zugrunde liegendes 3D-Objekt unterstützt.
    """
    def __init__(self, name : str):
        """
        Initialisiert ein kinematisches Kettenglied mit Namen, 3D-Darstellung und Koordinatenrahmen.

        Jedes Element kann Kinder und ein übergeordnetes Element haben, sowie ein Render-Objekt
        für die Visualisierung und einen lokalen Koordinatenrahmen (Frame).

        :param name: Name des kinematischen Kettenglieds.
        """

        self.name:str = name
        self.children:list[Kinematic_Chain_Element] = []
        self.parent:Kinematic_Chain_Element|None = None
        self.renderable:Object3D = three.Group()
        self.frame:Object3D = util.create_axes(0.3, show_labels=False, arrow_size=0.1, transparent_arrows=False)
        self.frame.visible = False






    def get_renderable(self) -> Object3D:
        """
        Gibt das zugrunde liegende renderbare 3D-Objekt zurück.

        :return: Das Object3D, das von diesem Element dargestellt wird.
        """
        return self.renderable






    def add(self, object:Object3D|Kinematic_Chain_Element) -> None:
        """
        Fügt ein Objekt oder Kettenelement als Kind hinzu und aktualisiert die Parent-Beziehung.

        :param object: Das hinzuzufügende Object3D oder Kinematic_Chain_Element.
        """
        renderable = object
        if hasattr(object, "get_renderable"):
            renderable = object.get_renderable()
        self.renderable.add(renderable)
        self.children.append(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = self





    def get_child_by_name(self, name:str) -> Kinematic_Chain_Element|None:
        """
        Gibt das erste Kindobjekt mit dem angegebenen Namen zurück.

        :param name: Name des gesuchten Kindes.
        :return: Das passende Kinematic_Chain_Element oder None, falls kein Kind gefunden wurde.
        """
        for c in self.children:
            if c.name == name:
                return c
        return None





    def get_matching_children(self, regex:str) -> list[Kinematic_Chain_Element]:
        """
        Gibt alle Kinderobjekte zurück, deren Namen einem gegebenen regulären Ausdruck entsprechen.

        :param regex: Regulärer Ausdruck zum Abgleichen der Kindernamen.
        :return: Liste der passenden Kinematic_Chain_Element-Objekte.
        """

        result:list[Kinematic_Chain_Element] = []
        for c in self.children:
            if re.fullmatch(regex, c.name):
                result.append(c)
        return result





    def remove(self, object) -> None:
        """
        Entfernt ein Kindobjekt aus der aktuellen Hierarchie und aktualisiert die Render- und Parent-Beziehungen.

        :param object: Das zu entfernende Objekt.
        """
        self.children.remove(object)
        if hasattr(object, "get_renderable"):
            self.renderable.remove(object.get_renderable())
        elif isinstance(object, Object3D):
            self.renderable.remove(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = None






    def show_frame(self, show:bool = True) -> None:
        """
        Zeigt oder versteckt das Koordinatensystem des Objekts.

        :param show: True, um das Koordinatensystem anzuzeigen; False, um es zu verstecken.
        """
        self.frame.visible=show





    def set_position(self, vec:np.ndarray):
        """
        Setzt die Position des Objekts auf den angegebenen Vektor.

        :param vec: Ein numpy-Array mit den neuen Koordinaten [X, Y, Z].
        """
        util.set_translation(self.renderable, vec)





    def get_position(self):
        """
        Gibt die aktuelle Position des Objekts zurück.

        :return: Ein Vektor oder Array mit den Koordinaten [X, Y, Z].
        """
        return self.renderable.position





    def set_rotation(self, vec_degree): 
        """
        Setzt die Rotation des Objekts anhand eines Euler-Winkel-Vektors.

        Die Euler-Winkel müssen in Grad angegeben werden. Intern wird die Reihenfolge "ZYX" verwendet.

        :param vec_degree: Iterable mit 3 Werten [X, Y, Z] für die Rotation in Grad.
        """
        util.set_rotation(self.renderable, vec_degree, "ZYX")




    def get_rotation(self, degrees=True) -> np.ndarray:   #INKONSISTENZ MIT DEM KOMPLETTEN REST DER API WEGEN DER REIHENFOLGE DES RÜCKGABEVEKTORS
        """
        Gibt die Rotation des Objekts als Euler-Winkel zurück.

        Die Euler-Winkel werden aus der Quaternion des Renderable-Objekts berechnet.
        Die Reihenfolge der zurückgegebenen Achsen ist immer x, y, z.

        :param degrees: Wenn True, werden die Winkel in Grad zurückgegeben, sonst in Radiant.
        :return: numpy-Array mit 3 Elementen: [X, Y, Z]-Rotation
        """
        q = self.renderable.quaternion  # Quaternion: [x, y, z, w]
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        euler_deg = r.as_euler("ZYX", degrees=degrees)
        return np.array(euler_deg.tolist()[::-1])
    



    def get_quaternion(self):
        """
        Gibt die aktuelle Orientierung als Quaternion zurück.

        Ruft die Quaternion aus dem Renderable-Objekt ab und gibt sie als Liste der Form [x, y, z, w] zurück.

        :return: Liste mit 4 Elementen, die die Quaternion (x, y, z, w) darstellen.
        """
        return list(self.get_renderable().quaternion)
    



    def get_rotvec(self):
        """
        Gibt den Rotationsvektor (Axis-Angle-Darstellung) des aktuellen Zustands zurück.

        Verwendet die aktuelle Orientierung als Quaternion und wandelt sie in einen Rotationsvektor
        um. Der Rotationsvektor ist eine 3D-Vektor-Darstellung der Rotation, bei der die Richtung
        die Rotationsachse angibt und die Länge des Vektors den Rotationswinkel in Radiant repräsentiert.

        :return: NumPy-Array der Form (3,) mit dem Rotationsvektor.
        """
        return R.from_quat(self.get_quaternion()).as_rotvec()
        











#----------------------------------------------------------------------------------------------------------------------







class Tool(Kinematic_Chain_Element):
    """
    Repräsentiert ein Werkzeug oder Endeffektor eines Roboters und kapselt dessen kinematische Struktur.

    Die Klasse lädt das URDF/Xacro-Modell des Werkzeugs, initialisiert alle Links und Gelenke und
    erstellt die zugehörigen 3D-Objekte. Sie erlaubt das Anzeigen oder Verstecken von Koordinatensystemen,
    das Setzen von Transparenz und die Abfrage einzelner Links oder Gelenke nach Namen.

    Besonderheiten:
        - Korrektur modellabhängiger Abweichungen für bestimmte Werkzeuge (z.B. Robotiq Greifer).
        - Unterstützung von Mimic-Joints.
        - Automatisches Laden von Meshes mit Material- und Farbinformationen.
        - Einheitliche Handhabung von Position (xyz), Orientierung (RPY) und Skalierung der Links.
    """
    def __init__(self, name:str):
        """
        Initialisiert ein Tool als kinematisches Kettenglied mit allen zugehörigen Links und Gelenken.

        Lädt die URDF-Daten des Tools basierend auf dem Namen, erstellt die Link- und Joint-Objekte
        und baut die 3D-Darstellung auf. Korrigiert spezifische Gelenkpositionen für bekannte Modelle.

        :param name: Name des Tools, der zur Suche der URDF/XACRO-Datei verwendet wird.
        """

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
        

        self._correct_elements()
        


    def _correct_elements(self):
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
        """
        Setzt die Transparenz aller Links und Gelenke des Roboters.

        :param opacity: Ein Wert zwischen 0 (vollständig transparent) und 1 (vollständig undurchsichtig).
        """
        for c in self.links:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity
        for c in self.joints:
            if hasattr(c.get_renderable(), "material"):
                c.get_renderable().material.transparent = True
                c.get_renderable().material.opacity = opacity





    def get_link_by_name(self, name:str) -> Link:
        """
        Gibt das Link-Objekt mit dem angegebenen Namen zurück.

        :param name: Name des gesuchten Links.
        :return: Das passende Link-Objekt oder None, falls kein Link gefunden wurde.
        """

        for l in self.links:
            if l.name == name:
                return l
        return None





    def get_joint_by_name(self, name:str) -> Joint:
        """
        Gibt das erste Gelenk mit dem angegebenen Namen zurück.

        :param name: Name des gesuchten Gelenks.
        :return: Das passende Joint-Objekt oder None, falls kein Gelenk gefunden wurde.
        """
        for j in self.joints:
            if j.name == name:
                return j
        return None
    



    def show_joint_frames(self, show=True):
        """
        Zeigt oder versteckt die Koordinatensysteme aller Gelenke des Roboters.

        :param show: True, um die Frames anzuzeigen; False, um sie zu verstecken.
        """
        for j in self.joints:
            j.frame.visible=show




    def show_link_frames(self, show=True):
        """
        Zeigt oder versteckt die Koordinatensysteme aller Links des Roboters.

        :param show: True, um die Koordinatensysteme anzuzeigen; False, um sie zu verstecken.
        """
        for l in self.links:
            l.frame.visible=show






    def init_links(self):
        """
        Initialisiert alle Links aus dem URDF-Dictionary und erstellt die zugehörigen 3D-Objekte.

        Für jeden Link werden Position, Orientierung (RPY), Geometrie, Skalierung und Materialinformationen
        ausgelesen. Abhängig von den Visualisierungsdaten wird entweder ein Mesh geladen oder ein leeres
        3D-Group-Objekt erstellt. Die erstellten Link-Objekte werden in die interne Link-Liste eingefügt.
        """
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
                            meshi = manager.load_mesh_auto_compatibility(mesh_path, hex_color, rgba[3], robot_name=self.name)
                    else:
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, link_element["visual"][0]["material"]["name"], robot_name=self.name)
                else:
                    meshi = manager.load_mesh_auto_compatibility(mesh_path, robot_name=self.name)
            else:
                meshi = three.Group()
            if scale is not None:
                meshi.scale=scale
                scale = [1,1,1]
            l = Link(link_element["name"], meshi, xyz, rpy)
            self.links.append(l)







    def init_joints(self):
        """
        Initialisiert alle Gelenke aus dem URDF-Dictionary und erstellt die kinematische Hierarchie.

        Für jedes Gelenk werden Position, Orientierung (RPY), Achse, Limits und eventuell Mimic-Beziehungen
        ausgelesen. Die Gelenke werden den entsprechenden Eltern- und Kind-Links zugeordnet und in die 
        interne Gelenkliste eingefügt.
        """
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
    """
    Repräsentiert ein Gelenk in einer kinematischen Kette mit Achse, Position, Rotation
    und optionalen Bewegungsgrenzen. Unterstützt die Nachahmung von Bewegungen durch
    andere Gelenke (Mimicer) und stellt eine 3D-Visualisierung über Three.js-Objekte bereit.

    Attribute:
    - axis (np.ndarray): Bewegungsachse des Gelenks.
    - position (np.ndarray): Position des Gelenks im Raum.
    - rotation (np.ndarray): Anfangsrotation des Gelenks in Grad.
    - lower_limit (float|None): Untere Bewegungsgrenze.
    - upper_limit (float|None): Obere Bewegungsgrenze.
    - is_mimicer (bool): Gibt an, ob dieses Gelenk ein Nachahmer ist.
    - mimicers (list[tuple[Kinematic_Chain_Element, float]]): Liste von Gelenken, die dieses
    Gelenk nachahmen, jeweils mit Multiplikator.
    - renderable (Object3D): 3D-Darstellung des Gelenks.
    - dh_frame (Object3D): Den Denavit-Hartenberg-Rahmen für Visualisierungen.
    - dh_alignment (np.ndarray): DH-Transformationsmatrix (4x4) für kinematische Berechnungen.
    """


    def __init__(self, name:str, axis:np.ndarray, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0]), lower_limit:float|None = None, upper_limit:float|None = None):
        """
        Initialisiert ein Gelenk mit Name, Achse, Position, Rotation und optionalen Bewegungsgrenzen.

        Erstellt die 3D-Darstellung des Gelenks, fügt den Standard-Rahmen hinzu und initialisiert
        den Denavit-Hartenberg-Rahmen für Visualisierung und kinematische Berechnungen.

        :param name: Name des Gelenks.
        :param axis: Bewegungsachse des Gelenks als 3D-Vektor.
        :param position: Startposition des Gelenks im Raum (Standard: [0,0,0]).
        :param rotation: Startrotation des Gelenks in Grad (Standard: [0,0,0]).
        :param lower_limit: Untere Bewegungsgrenze (optional).
        :param upper_limit: Obere Bewegungsgrenze (optional).
        """
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
        """
        Fügt ein Element hinzu, das dieses Gelenk „mimict“ (nachahmt) und optional mit einem
        Multiplikator skaliert.

        Das hinzugefügte Element muss ein Joint sein. Der Multiplikator wird beim Nachahmen
        des Gelenkwinkels angewendet.

        :param mimicer_and_multiplier: Ein Tupel aus (Kinematic_Chain_Element, float), wobei das
                                    Element das nachahmende Gelenk ist und der Float den
                                    Multiplikator angibt.
        """

        if not isinstance(mimicer_and_multiplier[0], Joint) : raise RuntimeError(f"jetzt ist klar, dass nicht nur Joints Mimicer sein können, sondern auch {type(mimicer_and_multiplier[0])}")
        mimicer_and_multiplier[0].is_mimicer = True
        self.mimicers.append(mimicer_and_multiplier)
    






    def get_previous_joint_in_chain(self) -> Joint|None:
        """
        Gibt das vorherige Gelenk in der kinematischen Kette zurück.

        Durchläuft die Eltern-Hierarchie des aktuellen Elements nach oben, bis das erste
        übergeordnete Gelenk gefunden wird. Gibt None zurück, falls kein solches Gelenk existiert.

        :return: Das erste gefundene übergeordnete Joint-Objekt oder None.
        """
        current = self.parent
        while current is not None:
            if isinstance(current, Joint):
                return current
            current = current.parent
        return current




class Link(Kinematic_Chain_Element):
    """
    Repräsentiert ein starr verbundenes Glied in einer kinematischen Kette.

    Jedes Link-Objekt hat eine zugeordnete 3D-Mesh-Darstellung und kann eine
    Position sowie Rotation im Raum besitzen. Das interne Achsen-Frame wird
    automatisch hinzugefügt.

    :param name: Name des Links.
    :param mesh: 3D-Mesh-Objekt, das das Link visuell repräsentiert.
    :param position: Startposition des Links im Raum (Standard: [0, 0, 0]).
    :param rotation: Startrotation des Links im Raum (Standard: [0, 0, 0]).
    """

    def __init__(self, name:str, mesh:Mesh, position:np.ndarray = np.array([0,0,0]), rotation:np.ndarray = np.array([0,0,0])) -> None:
        """
        Initialisiert ein Link-Objekt mit Name, Mesh, Position und Rotation.

        Fügt das interne Achsen-Frame dem Mesh hinzu und setzt die Startposition
        und -rotation des Links im Raum.

        :param name: Name des Links.
        :param mesh: 3D-Mesh-Objekt, das das Link visuell darstellt.
        :param position: Startposition des Links als numpy-Array (Standard: [0,0,0]).
        :param rotation: Startrotation des Links als numpy-Array (Standard: [0,0,0]).
        """
        super().__init__(name)
        self.renderable:Mesh = mesh
        self.renderable.add(self.frame)
        self.set_position(position)
        self.set_rotation(rotation)
        
    

        










#---------------------------------------------MANIPULATOR-------------------------------------------------------------------------









class Manipulator:
    """
    Repräsentiert einen kinematischen Manipulator (Roboterarm) basierend auf URDF-/XACRO-Beschreibungen.  

    Die Klasse verwaltet Links, Joints, optionale Werkzeuge und die Kinematik des Roboters.  
    Sie ermöglicht den Zugriff auf die interne Struktur, das Laden und Speichern von Posen, 
    sowie die Integration in eine Umgebung.  

    Ein `Manipulator` bildet die zentrale Schnittstelle zur Simulation, Analyse und Steuerung 
    eines Roboterarms.  
    """

    def __init__(self, name:str, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        """
        Erzeugt eine neue Manipulator-Instanz basierend auf URDF-/XACRO-Beschreibungen.  

        Die Klasse initialisiert Links, Joints, optionale Werkzeuge (z. B. Greifer) sowie
        die zugehörigen kinematischen Strukturen. Zusätzlich werden die Basis- und Tool-Verbindungen
        automatisch erkannt und gesetzt.  

        :param name: Name des Roboters, wird zur Lokalisierung der XACRO-/URDF-Dateien genutzt.  
        :param tool_name: Name des Werkzeuges (z. B. Greifer), das am TCP montiert wird. Standard: "robotiq_arg2f_140_model".  
        :param position: Anfangsposition des Basis-Links im Weltkoordinatensystem.
        :raises AssertionError: Falls `base_link` oder `base` nicht in der URDF-Struktur gefunden werden.
        """
        self.k0 = None
        #self.inspector = None
        self.environment = None
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
        
        #print(urdf)
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
        self.base_link.set_position(position)







    def set_environment(self, env:util.Environment):
        """
        Setzt die Umgebungsreferenz für das aktuelle Robotermodell.  

        Dies erlaubt es, den Roboter in einer bestimmten Umgebung zu verankern 
        und auf deren Objekte oder Konfigurationen zuzugreifen.

        :param env: Instanz der Umgebungsklasse (:class:`util.Environment`).
        """
        self.environment = env








    def learn(self, pose_name: str, thetas:list|None = None):
        """
        Speichert eine neue Pose des Roboters unter einem angegebenen Namen.

        Überprüft zunächst, ob bereits eine Pose mit diesem Namen existiert, und
        bricht ggf. mit einem Fehler ab. Falls keine Gelenkwinkel (`thetas`) 
        übergeben wurden, werden diese aus dem aktuellen Zustand der DH-Kette 
        ermittelt. Die neue Pose wird anschließend in einer JSON-Datei 
        gespeichert und die interne Pose-Liste aktualisiert.

        :param pose_name: Eindeutiger Name der zu speichernden Pose.
        :param thetas: Optionale Liste von Gelenkwinkeln. Falls None, wird der
                    aktuelle Zustand aus den DH-Parametern übernommen.
        :raises RuntimeError: Falls bereits eine Pose mit gleichem Namen existiert.
        """
        display(f"POSEN: {self.learned_poses}")
        exists = any(obj["name"] == pose_name for obj in self.learned_poses)
        if exists : raise RuntimeError("Pose mit diesem Namen existiert bereits")
        filepath = f"{manager.teach_path}/{self.name}.json"
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
        """
        Lädt die gespeicherten Posen des Roboters aus einer JSON-Datei.

        Überprüft, ob eine entsprechende Datei existiert, und lädt deren Inhalt.
        Falls die Datei beschädigt oder leer ist, wird eine Warnung ausgegeben
        und eine leere Liste zurückgegeben. Falls keine Datei existiert, wird
        ebenfalls eine leere Liste zurückgegeben.

        :return: Liste der gespeicherten Posen oder eine leere Liste.
        """
        filepath = f"{manager.teach_path}/{self.name}.json"
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
        """
        Gibt die Gelenkwinkel einer gespeicherten Pose anhand ihres Namens zurück.

        Durchsucht die gespeicherten Posen nach einer Pose mit dem angegebenen Namen
        und liefert deren Gelenkwinkel (Theta-Werte) zurück. 

        :param name: Der Name der gesuchten Pose.
        :return: Die Gelenkwinkel (Theta-Werte) der gefundenen Pose.
        :raises ValueError: Falls keine Pose mit dem angegebenen Namen existiert.
        """
        for pose in self.learned_poses:
            if pose["name"] == name:
                return pose["theta"]
        raise ValueError(f"Pose '{name}' nicht gefunden.")







    def update_dh_angles(self):
        """
        Aktualisiert die DH-Gelenkwinkel basierend auf den aktuellen Joint-Konfigurationen.

        Für jedes Gelenk im DH-Modell wird der zugehörige Joint gesucht und dessen 
        Rotationsvektor ausgewertet. Aus dessen Norm wird der neue Gelenkwinkel berechnet 
        und unter Berücksichtigung der Achsorientierung (Vorzeichenkorrektur) im 
        DH-Modell aktualisiert.

        :raises RuntimeError: Falls ein Joint nicht gefunden werden kann oder 
                            keine gültige Achse definiert ist.
        """
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
        """
        Berechnet und gibt die globale Transformationsmatrix des TCP (Tool Center Point) zurück.

        Aktualisiert zunächst die Gelenkwinkel im DH-Modell und führt anschließend
        die Vorwärtstransformation (Forward Kinematics) aus.

        :return: 4x4-Homogenmatrix (numpy.ndarray), die die globale Pose des TCP beschreibt.
        """
        self.update_dh_angles()
        return self.dh.forward_kinematics()






    def animate_stable(self, joints:list, angles_rad:list, duration:float = 2) -> None:
        """
        Führt eine stabile Animation mehrerer Gelenke über eine gegebene Dauer aus.  

        Die Gelenkwinkel werden animiert, anschließend werden die finalen Orientierungen 
        der Render-Objekte konsistent gesetzt, um numerische Fehler oder Abweichungen 
        aus der Animation zu korrigieren.  

        :param joints: Liste der zu animierenden Gelenke.
        :param angles_rad: Liste der Zielwinkel (in Radiant) für die Gelenke.
        :param duration: Gesamtdauer der Animation in Sekunden (Standard: 2).
        """  
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
        """
        Animiert den Manipulator anhand einer zuvor gelernten Pose.  

        Die Pose wird über ihren Namen geladen und anschließend mit der internen 
        `animate_by_theta`-Methode abgespielt.  

        :param name: Name der gelernten Pose.  
        :param duation: Gesamtdauer der Animation in Sekunden (Standard: 1.0).  
        :param quality: Qualitätsfaktor der Animation (Standard: 1.0).  
        :param synchronous: Falls True, blockiert die Ausführung bis zum Ende der Animation.  
        :param with_tcp_target: Falls True, wird zusätzlich das TCP-Ziel animiert (Standard: True).  
        """  
        pose = self.get_learned_pose(name)
        self.animate_by_theta(pose, duation, quality, synchronous, with_tcp_target)
        






    def animate_by_theta(self, theta:list, duration:float = 1.0, quality:float = 1.0, synchronous:bool = False, with_tcp_target:bool = True):
        """
        Animiert den Manipulator gemäß einer gegebenen Liste von Gelenkwinkeln.

        Die Gelenkwinkel werden auf die jeweiligen Gelenke in der Kette angewendet. 
        Dabei wird die Orientierung der DH-Achsen berücksichtigt, um die korrekte Richtung
        der Rotation sicherzustellen.

        :param theta: Liste der Gelenkwinkel in Radiant, in der Reihenfolge der DH-Joints.  
        :param duration: Gesamtdauer der Animation in Sekunden (Standard: 1.0).  
        :param quality: Qualitätsfaktor der Animation (Standard: 1.0).  
        :param synchronous: Falls True, blockiert die Ausführung bis zum Ende der Animation.  
        :param with_tcp_target: Falls True, wird zusätzlich das TCP-Ziel animiert (Standard: True).  
        """
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
        """
        Animiert die angegebenen Gelenke auf die Zielwinkel über quaternion-basierte SLERP-Interpolation.

        Unterstützt rekursive Mimic-Gelenke, optionales Animieren des TCP-Ziels und die Wahl zwischen
        relativer Rotation zur Basis oder absoluter Rotation. Kann synchron oder asynchron ausgeführt werden.

        :param joints: Liste der zu animierenden Joint-Objekte.
        :param angles_rad: Liste der Zielwinkel in Radiant für jedes Gelenk.
        :param duration: Gesamtdauer der Animation in Sekunden (Standard: 1.0).
        :param quality: Qualitätsfaktor der Animation, beeinflusst Glätte/Framerate (Standard: 1.0).
        :param add_on_baserotation: Falls True, wird Rotation relativ zur aktuellen Basisrotation durchgeführt.
        :param synchronous: Falls True, blockiert die Ausführung bis zum Ende der Animation.
        :param with_tcp_target: Falls True, wird zusätzlich das TCP-Ziel animiert (Standard: True).
        """
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

            fps_start = time.perf_counter()
            while(t < duration):
                start = time.perf_counter()
                block(t/duration, self.base_link, step)
                time.sleep(0.001/quality)
                end = time.perf_counter()
                t += end - start
                fps_end = time.perf_counter()
                fps_diff = fps_end - fps_start
                fps_start = fps_end
                #fps_text.value= f"{(1/fps_diff)}"
            block(1, self.base_link, step)

            if self.environment is not None and with_tcp_target:
                util.apply_transformation_matrix(self.tcp_target, self.get_global_tcp_transform())
                self.environment.add(self.tcp_target) 


        thread = threading.Thread(target=_animate_experimental_task, args=(joints, angles_rad, duration, quality))
        thread.start()
        if synchronous : thread.join()  















    def apply_DH_model(self, dh:DHKinematicModel) -> None:
        """
        Wendet ein DH-Kinematikmodell auf den Manipulator an und initialisiert die DH-Funktionen.

        Berechnet die Transformationsmatrizen für jedes Gelenk, setzt die DH-Ausrichtung und erstellt
        die zugehörigen DH-Rahmen (Frames). Zusätzlich werden die Achsen für den Basisrahmen (k0)
        und das TCP-Ziel erstellt und initial transformiert.

        :param dh: DH-Kinematikmodell, das auf den Manipulator angewendet werden soll.
        """
        self.dh = dh
        self.dh.init_functions()
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
        self.tcp_target = util.create_axes(0.3, 0.1, False, "", 0.2)
        util.apply_transformation_matrix(self.tcp_target, self.get_global_tcp_transform())
        #self.environment.add(self.tcp_target)





        




    

    def init_links(self) -> None:
        """
        Initialisiert alle Links des Manipulators basierend auf der URDF-Beschreibung.

        Für jedes Link-Element werden Position, Rotation und Mesh aus der URDF-Datei ausgelesen.
        Falls Materialinformationen vorhanden sind, werden diese ebenfalls berücksichtigt. Die
        Links werden anschließend in der `self.links`-Liste gespeichert. Die Initialisierung
        erfolgt multithreaded, um die Ladezeit bei vielen Links zu reduzieren.
        """
        self.links = [None] * len(self.urdf_dictionary["links"])
        threads = []

        def worker(index, link_element):
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
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, hex_color, rgba[3], robot_name=self.name)
                    else:
                        meshi = manager.load_mesh_auto_compatibility(mesh_path, link_element["visual"][0]["material"]["name"], robot_name=self.name)
                else:
                    meshi = manager.load_mesh_auto_compatibility(mesh_path, robot_name=self.name)
            else:
                meshi = three.Group()
            l = Link(link_element["name"], meshi, xyz, rpy)
            self.links[index] = l


        i = 0
        for link_element in self.urdf_dictionary["links"]:
            t = threading.Thread(target=worker, args=(i, link_element))    
            t.start()
            threads.append(t)
            i+=1

        for t in threads:
            t.join()






    def init_joints(self) -> None:
        """
        Initialisiert alle Gelenke des Manipulators basierend auf der URDF-Beschreibung.

        Für jedes Gelenk werden Position, Rotation, Achse sowie eventuelle Bewegungsgrenzen
        aus der URDF-Datei ausgelesen. Eltern- und Kind-Links werden verknüpft, und falls
        ein Gelenk ein Mimic-Gelenk ist, wird der entsprechende Multiplikator berücksichtigt.
        Die erstellten Joint-Objekte werden in `self.joints` gespeichert.
        """
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
        """
        Zeigt oder versteckt die lokalen Referenzkoordinatensysteme aller Gelenke des Manipulators.

        :param show: Falls True, werden die Gelenk-Koordinatensysteme sichtbar gemacht; 
                    falls False, werden sie ausgeblendet (Standard: True).
        """
        for j in self.joints:
            j.frame.visible=show








    def show_link_frames(self, show=True):
        """
        Zeigt oder versteckt die lokalen Referenzrahmen aller Links des Manipulators.

        :param show: Falls True, werden die Link-Frames sichtbar gemacht; 
                    falls False, werden sie ausgeblendet (Standard: True).
        """
        for l in self.links:
            l.frame.visible=show








    def show_DH_frames(self, show=True):
        """
        Zeigt oder versteckt die DH-Referenzkoordinatensysteme aller Gelenke des Manipulators.

        Die DH-Referenzkoordinatensysteme werden entlang der Kette vom Basislink bis zu den Endeffektoren aktualisiert
        und korrekt positioniert, um die aktuelle DH-Ausrichtung zu visualisieren.

        :param show: Falls True, werden die DH-Frames sichtbar gemacht; 
                    falls False, werden sie ausgeblendet (Standard: True).
        """
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
        """
        Liefert das 3D-Renderable-Objekt des Manipulators zurück.

        :return: Das Haupt-Mesh-Objekt, das den Manipulator visuell repräsentiert.
        """
        return self.mesh
    






    def get_link_by_name(self, name:str) -> Link|None:
        """
        Sucht einen Link anhand seines Namens im Manipulator.

        :param name: Name des gesuchten Links.
        :return: Das Link-Objekt, falls gefunden, sonst None.
        """
        for l in self.links:
            if l.name == name:
                return l
        return None






    def get_joint_by_name(self, name:str) -> Joint|None:
        """
        Sucht ein Gelenk anhand seines Namens im Manipulator.

        :param name: Name des gesuchten Gelenks.
        :return: Das Joint-Objekt, falls gefunden, sonst None.
        """
        for j in self.joints:
            if j.name == name:
                return j
        return None
    






    def get_matching_joints(self, regex:str) -> list[Joint]:
        """
        Liefert eine Liste aller Gelenke, deren Name einem gegebenen regulären Ausdruck entspricht.

        :param regex: Regulärer Ausdruck zum Abgleich der Gelenknamen.
        :return: Liste der übereinstimmenden Joint-Objekte.
        """
        result : list[Joint] = []
        for j in self.joints:
            if re.fullmatch(regex, j.name):
                result.append(j)
        return result
    

    def get_matching_links(self, regex:str) -> list[Link]:
        """
        Liefert eine Liste aller Links, deren Name einem gegebenen regulären Ausdruck entspricht.

        :param regex: Regulärer Ausdruck zum Abgleich der Linknamen.
        :return: Liste der übereinstimmenden Link-Objekte.
        """
        result : list[Link] = []
        for l in self.links:
            if re.fullmatch(regex, l.name):
                result.append(l)
        return result

    



    def find_link_to_tool_joint(self):
        """
        Findet das Gelenk, das den Manipulator mit dem Tool verbindet.

        :return: Joint-Objekt, das mit dem Link "tool0" verbunden ist.
        """
        return self.get_link_by_name("tool0").parent
        
        




    def find_base_link_to_link_1_joint(self) -> Joint:
        """
        Findet das Gelenk, das den Basis-Link mit dem ersten Link verbindet.

        Es wird geprüft, dass genau ein passendes Gelenk existiert und dass 
        dieses tatsächlich vom Typ Joint ist. Andernfalls wird ein Fehler ausgelöst.

        :return: Joint-Objekt, das den Basis-Link mit Link 1 verbindet.
        :raises AssertionError: Wenn mehr als ein passendes Gelenk gefunden wurde.  
        :raises TypeError: Wenn das gefundene Kind kein Joint ist.  
        """
        matching_base_link_children = self.base_link.get_matching_children(".*joint.*1.*")
        if len(matching_base_link_children) != 1:
            raise AssertionError("es wurde mehr als ein joint vom base_link zu link_1 gefunden")
        elif not isinstance(matching_base_link_children[0], Joint):
            raise TypeError(f"statt joint eins wurde hier {matching_base_link_children[0].name} gefunden von der Klasse {matching_base_link_children[0].__class__}")
        return matching_base_link_children[0]






    def set_opacity(self, opacity:float, tool_also:bool = False) -> None:
        """
        Setzt die Transparenz (Opacity) des Manipulators.

        Alle Links und Gelenke mit Materialeigenschaften werden entsprechend 
        angepasst. Optional kann auch das Tool des Manipulators einbezogen werden.

        :param opacity: Wert für die Transparenz (0.0 = unsichtbar, 1.0 = vollständig sichtbar).  
        :param tool_also: Falls True, wird zusätzlich die Transparenz des Tools gesetzt (Standard: False).  
        """
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
        """
        Gibt die Namen aller Links des Manipulators auf der Konsole aus.

        :return: None  
        """
        print("\nLinks:")
        for link in self.links:
            print(link.name)
        print()




    def print_joints(self) -> None:
        """
        Gibt die Namen aller Joints des Manipulators auf der Konsole aus.

        :return: None  
        """
        print("\nJoints:")
        for joint in self.joints:
            print(joint.name)
        print()




    def print_kinematic_chain(self) -> None:
        """
        Gibt die kinematische Kette des Manipulators von der Basis bis zum Endeffektor auf der Konsole aus.

        Die Ausgabe erfolgt als sequenzielle Liste der verbundenen Links und Joints.  

        :return: None  
        """
        print("\nKinematic Chain:")
        current = self.links[0]
        print(current.name)
        while(len(current.children) > 0):
            current = current.children[0]
            print(current.name)
        print()





    def compute_global_transform(self, current:Kinematic_Chain_Element = None, parent_transform:np.ndarray = np.eye(4), global_transforms:dict = None, with_print:bool = False):
        """
        Berechnet die globalen Transformationsmatrizen für alle Elemente der kinematischen Kette.  

        Ausgehend vom Basis-Link wird rekursiv die Transformation jedes Elements (Links oder Joints) 
        bestimmt, indem die lokale Pose mit der Transformationsmatrix des Elternteils kombiniert wird.  
        Die Ergebnisse werden in einem Dictionary gespeichert, das jedem Elementnamen seine globale 
        Transformationsmatrix zuordnet.  

        :param current: Aktuelles Kettenelement, von dem aus die Berechnung gestartet wird (Standard: base_link).  
        :param parent_transform: Transformationsmatrix des Elternteils (Standard: Einheitsmatrix).  
        :param global_transforms: Dictionary zur Speicherung der Ergebnisse (Standard: neues Dictionary).  
        :param with_print: Falls True, werden die Transformationen zusätzlich auf der Konsole ausgegeben.  
        :return: Dictionary mit den globalen Transformationsmatrizen aller Kettenglieder.  
        """
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



class lbr_iiwa_14_r820(Manipulator):
    """
    Klasse zur Modellierung des KUKA LBR iiwa 14 R820 Manipulators.  

    Diese Klasse erbt von :class:`Manipulator` und definiert die kinematischen 
    Eigenschaften des Roboters mithilfe des Denavit-Hartenberg-Modells (DH-Parameter).  
    Sie ermöglicht die Initialisierung des Roboters mit Werkzeug und Basisposition 
    sowie die Anwendung der DH-Kinematik auf die Gelenkkette.  
    """

    def __init__(self, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        """
        Konstruktor zur Initialisierung des KUKA LBR iiwa 14 R820 Manipulators.  

        Es werden die DH-Parameter für alle Gelenke definiert und daraus ein 
        DH-Kinematikmodell erzeugt. Dieses Modell wird anschließend auf die interne 
        Repräsentation des Manipulators angewendet, sodass die vollständige kinematische 
        Kette korrekt aufgesetzt ist.  

        :param tool_name: Name des am TCP montierten Werkzeugs (Standard: "robotiq_arg2f_140_model").  
        :param position: Startposition des Roboters im Raum als Vektor [x, y, z] (Standard: [0,0,0]).  
        """
        super().__init__(name="lbr_iiwa_14_r820", tool_name=tool_name, position=position)
        dh_parameters : dict= {}
        dh_parameters[self.joints[1].name] = {"theta":0,        "d":0.36,      "a":-0.00043624,   "alpha":np.pi/2}
        dh_parameters[self.joints[2].name] = {"theta":0,        "d":0,         "a":0,             "alpha":-np.pi/2}
        dh_parameters[self.joints[3].name] = {"theta":0,        "d":0.42,      "a":0.00043624,    "alpha":np.pi/2}
        dh_parameters[self.joints[4].name] = {"theta":0,        "d":0,         "a":0,             "alpha":-np.pi/2}
        dh_parameters[self.joints[5].name] = {"theta":0,        "d":0.4,       "a":0,             "alpha":np.pi/2}
        dh_parameters[self.joints[6].name] = {"theta":0,        "d":0.0,       "a":0,             "alpha":-np.pi/2}
        dh_parameters[self.joints[7].name] = {"theta":0,        "d":0.126,       "a":0,           "alpha":0}
        #base_transform = pose_to_matrix(self.base_to_baselink_joint.get_position(), self.base_to_baselink_joint.get_rotation(False), False)
        #k0 = compute_dh_matrix(0.0, 0.0, 0.0, 0.0)
        DH = DHKinematicModel(dh_parameters=dh_parameters)
        #DH.base_to_dh = DHKinematicModel.compute_base_to_dh(base_transform, k0)
        DH.dh_to_tool = DH.compute_dh_to_tool(self.compute_global_transform()[self.link_to_tool_joint.name])
        self.apply_DH_model(DH)



class kr6r900_2(Manipulator):
    """
    Klasse zur Modellierung des KUKA KR6 R900-2 Manipulators.  

    Diese Klasse erbt von :class:`Manipulator` und definiert die kinematischen 
    Eigenschaften des Roboters über die Denavit-Hartenberg-Parameter (DH-Parameter).  
    Sie ermöglicht die Initialisierung mit einem Werkzeug und einer Basisposition 
    sowie die Anwendung der DH-Kinematik auf die Gelenkkette.  
    """

    def __init__(self, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        """
        Konstruktor zur Initialisierung des KUKA KR6 R900-2 Manipulators.  

        Es werden die DH-Parameter für die relevanten Gelenke definiert und ein 
        DH-Kinematikmodell erstellt, das anschließend auf die interne Darstellung 
        des Manipulators angewendet wird.  

        :param tool_name: Name des am TCP montierten Werkzeugs (Standard: "robotiq_arg2f_140_model").  
        :param position: Startposition des Roboters im Raum als Vektor [x, y, z] (Standard: [0,0,0]).  
        """
        super().__init__(name="kr6r900_2", tool_name=tool_name, position=position)
        dh_parameters : dict= {}
        dh_parameters[self.joints[1].name] = {"theta":0,        "d":0.4,    "a":0.025,   "alpha":np.pi/2}
        dh_parameters[self.joints[2].name] = {"theta":0,        "d":0,      "a":0.455,   "alpha":0}
        dh_parameters[self.joints[3].name] = {"theta":np.pi/2,  "d":0,      "a":0.025,   "alpha":np.pi/2}
        dh_parameters[self.joints[4].name] = {"theta":0,        "d":0.42,   "a":0,       "alpha":-np.pi/2}
        dh_parameters[self.joints[5].name] = {"theta":0,        "d":0,      "a":0,       "alpha":np.pi/2}
        dh_parameters[self.joints[7].name] = {"theta":0,        "d":0.09,   "a":0,       "alpha":0}    #7 weil es so im urdf geordnet ist...6 wäre der joint "base_link-base"
        #base_transform = pose_to_matrix(self.base_to_baselink_joint.get_position(), self.base_to_baselink_joint.get_rotation(False), False)
        #k0 = compute_dh_matrix(0.0, 0.0, 0.0, 0.0)
        DH = DHKinematicModel(dh_parameters=dh_parameters)
        #DH.base_to_dh = DHKinematicModel.compute_base_to_dh(base_transform, k0)
        DH.dh_to_tool = DH.compute_dh_to_tool(self.compute_global_transform()[self.link_to_tool_joint.name])
        self.apply_DH_model(DH)
        





class irb6640_185_280(Manipulator):
    """
    Klasse zur Modellierung des ABB IRB 6640-185/280 Manipulators.  

    Diese Klasse erbt von :class:`Manipulator` und definiert die kinematischen 
    Eigenschaften des Roboters über die Denavit-Hartenberg-Parameter (DH-Parameter).  
    Sie ermöglicht die Initialisierung mit einem Werkzeug und einer Basisposition 
    sowie die Anwendung der DH-Kinematik auf die Gelenkkette.  
    """

    def __init__(self, tool_name:str = "robotiq_arg2f_140_model", position:np.ndarray = np.array([0,0,0])):
        """
        Konstruktor zur Initialisierung des ABB IRB 6640-185/280 Manipulators.  

        Es werden die DH-Parameter für die relevanten Gelenke definiert und ein 
        DH-Kinematikmodell erstellt, das anschließend auf die interne Darstellung 
        des Manipulators angewendet wird.  

        :param tool_name: Name des am TCP montierten Werkzeugs (Standard: "robotiq_arg2f_140_model").  
        :param position: Startposition des Roboters im Raum als Vektor [x, y, z] (Standard: [0,0,0]).  
        """
        super().__init__(name="irb6640_185_280", tool_name=tool_name, position=position)
        dh_parameters : dict= {}
        dh_parameters[self.joints[1].name] = {"theta":0,        "d":0.78,      "a":0.32,   "alpha":np.pi/2}
        dh_parameters[self.joints[2].name] = {"theta":np.pi/2,  "d":0,      "a":1.075,  "alpha":0}
        dh_parameters[self.joints[3].name] = {"theta":0,        "d":0,      "a":0.2,    "alpha":np.pi/2}
        dh_parameters[self.joints[4].name] = {"theta":0,        "d":1.392,  "a":0,      "alpha":-np.pi/2}
        dh_parameters[self.joints[5].name] = {"theta":0,        "d":0,      "a":0,      "alpha":np.pi/2}
        dh_parameters[self.joints[6].name] = {"theta":0,        "d":0.2,    "a":0,      "alpha":0}
        #base_transform = pose_to_matrix(self.base_to_baselink_joint.get_position(), self.base_to_baselink_joint.get_rotation(False), False)
        #k0 = compute_dh_matrix(0.0, 0.0, 0.0, 0.0)
        DH = DHKinematicModel(dh_parameters=dh_parameters)
        #DH.base_to_dh = DHKinematicModel.compute_base_to_dh(base_transform, k0)
        DH.dh_to_tool = DH.compute_dh_to_tool(self.compute_global_transform()[self.link_to_tool_joint.name])
        self.apply_DH_model(DH)





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
    """
    Animiert die Rotation eines einzelnen Joints um eine gegebene Achse.  

    Die Funktion berechnet die Zielrotation anhand der Basisorientierung des Joints 
    und interpoliert diese über eine definierte Dauer mithilfe eines Quaternion-Keyframe-Tracks.  
    Optional wird die Animation als Endlosschleife ausgeführt.  
    Falls der Joint Mimiker hat, wird die Animation rekursiv auch auf diese angewendet.  

    :param joint: Joint, der animiert werden soll.  
    :param axis: Rotationsachse als Vektor [x, y, z].  
    :param angle_rad: Rotationswinkel in Radiant.  
    :param loop: Falls True, wird die Animation endlos wiederholt (Standard: False).  
    :param duration: Dauer der Animation in Sekunden (Standard: 2).  
    :return: Tuple bestehend aus (AnimationAction, Zielquaternion, Joint).  
    """
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








