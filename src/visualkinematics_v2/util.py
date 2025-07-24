from __future__ import annotations
import numpy as np
import sympy as sp
import pythreejs as three
from ipywidgets import *
from IPython.display import display
from pythreejs import *
from pythreejs import SpriteMaterial, Sprite
import time
from scipy.spatial.transform import Rotation as R, Slerp
import os
import visualkinematics_v2.manager as manager
from collections.abc import Iterable
import typing
from numba import njit
from typing import Callable
from contextlib import contextmanager
import threading
import ipyevents









def rot_axis_from_rot_mat(rot_mat):
    '''
    Gibt die Rotationsachse von rot_mat zurueck.

    :param rot_mat: Die Rotationsmatrix als mehrdimensionales Array. 

    :return: Rotationsachse als normalisierter Vektor z.B [x,y,z].
    '''
    if isinstance(rot_mat,(sp.Basic, sp.MatrixBase)):
        rot_mat.evalf()
    R = np.array(rot_mat)

    # Eigenwerte und Eigenvektoren berechnen
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # Eigenvektor für Eigenwert 1 finden
    axis = eigenvectors[:, np.isclose(eigenvalues, 1)]

    # Ergebnis auf 1D normieren (optional)
    axis = axis[:, 0]
    axis = axis / np.linalg.norm(axis)
    return np.real(axis)
   



def quaternion_to_euler(x, y, z, w, order="ZYZ"):
    '''
    Wandelt ein Quaternion in Eulerwinkel um. Dabei wird die Übergebene Rotationsreihenfolge für die Eulerwinkel beachtet.
    
    :param x: x-Komponente des Quaternions.
    :param y: y-Komponente des Quaternions.
    :param z: z-Komponente des Quaternions.
    :param w: w-Komponente des Quaternions.
    :param order: Rotationsreihenfolge für die Eulerwinkel als String.

    :return: Die Eulerwinkel als Array in der Reihenfolge, wie es order vorgibt, z.B order="ZXY rückgabe->[z,x,y].
    '''
    quaternion = [x, y, z, w]
    euler_angles = R.from_quat(quaternion).as_euler(order, degrees=True)
    return euler_angles




def euler_to_rot_mat(angles, order="ZYZ"):
    '''
    Wandelt Eulerwinkel in eine Rotationsmatrix um.

    :param angles: Die Eulerwinkel in Grad. Diese Müssen in der Reihenfolge angegeben werden wie es order vorgibt z.B angles=[y,x,z] order="YXZ".
    :param order: Rotationsreihenfolge für die Eulerwinkel als String.

    :return: Die Rotationsmatrix als mehrdimensionales Array.
    '''
    if isinstance(angles,(sp.Basic, sp.MatrixBase)):
        angles.evalf()
    r = R.from_euler(order, angles, degrees=True)
    return r.as_matrix()



def rot_matrix_to_euler(rot_mat, order="ZYZ"):
    '''
    Wandelt eine Rotationsmatrix in Eulerwinkel um.

    :param rot_mat: Die Rotationsmatrix.
    :param order: Rotationsreihenfolge für die Eulerwinkel als String. 

    :return: Die Eulerwinkel. Diese werden in der Reihenfolge zurueckgegeben, wie es order vorgibt z.B order="ZXY" rueckgabe->[Z,X,Y].
    '''
    if isinstance(rot_mat,(sp.Basic, sp.MatrixBase)):
        rot_mat.evalf()
    r = R.from_matrix(rot_mat)
    return r.as_euler(order, degrees=True)




def rot_matrix_to_quaternion(rot_mat):
    '''
    Wandelt eine Rotationsmatrix in ein Quaternion um.

    :param rot_mat: Die Rotationsmatrix als mehrdimensionales Array.

    :return: das Quaternion als Array.
    '''
    if isinstance(rot_mat,(sp.Basic, sp.MatrixBase)):
        rot_mat.evalf()
    r = R.from_matrix(rot_mat).as_quat()
    return r





def euler_to_quaternion(angles, order='ZYZ'):
    '''
    Wandelt Eulerwinkel in ein Quaternion um.

    :param angles: Die Eulerwinkel in Grad. Diese müssen in der Reihenfolge angegeben werden, wie es order vorgibt z.B angles=[y,x,z] order="YXZ".
    :param order: Rotationsreihenfolge für die Eulerwinkel als String.

    :return: Das Quaternion als Array.
    '''
    if isinstance(angles,(sp.Basic, sp.MatrixBase)):
        angles.evalf()
    r = R.from_euler(order, angles, degrees=True)
    quat = r.as_quat()
    return quat




def compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Berechnet Vertex-Normalen für ein Dreiecksnetz.

    :param vertices: (N, 3) array of float32 – die Eckpunkte
    :param indices: (M,) array of uint32 – flach (also dreifach pro Dreieck)
    :return: (N, 3) array of float32 – die berechneten Vertex-Normalen
    """
    num_vertices = vertices.shape[0]
    normals = np.zeros((num_vertices, 3), dtype=np.float32)

    triangles = indices.reshape(-1, 3)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Kantenvektoren
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Flächennormalen (cross product)
    face_normals = np.cross(edge1, edge2)

    # Normale zu jedem Vertex aufsummieren
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)

    # Normalisieren
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0  # Division durch 0 vermeiden
    normals /= lengths[:, np.newaxis]

    return normals




def order_angles(angles: list, from_order:str, to_order: str) -> list:
    '''
    Ordnet die übergebenen Eulerwinkel von einer Reihenfolge in eine andere.

    :param angles: Liste der Winkel im Format [a1, a2, a3] in from_order.
    :param from_order: Der aktuelle Rotationsachsen-String (z. B. "ZYX").
    :param to_order: Die gewünschte neue Reihenfolge der Achsen (z. B. "XYZ").
    :return: Liste der Winkel in der neuen Reihenfolge to_order.
    '''
    from_order = from_order.upper()
    to_order = to_order.upper()

    # Erstelle ein Mapping von Achsen zu Winkeln
    angle_map = {axis: angle for axis, angle in zip(from_order, angles)}

    # Baue die neue Liste der Winkel in der gewünschten Reihenfolge
    return [angle_map[axis] for axis in to_order]




def quaternion_multiply(q1, q2):
    '''
    Multipliziert zwei Quaternions.

    :param q1: Das erste Quaternion als Liste oder Array [x, y, z, w].
    :param q2: Das zweite Quaternion als Liste oder Array [x, y, z, w].

    :return: Das Ergebnis der Quaternion-Multiplikation als Liste [x, y, z, w].
    '''
    if isinstance(q1,(sp.Basic, sp.MatrixBase)):
        q1.evalf()
    if isinstance(q2,(sp.Basic, sp.MatrixBase)):
        q2.evalf()
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return [x, y, z, w]








def create_axes(len, font_scale=0.4, show_labels=True, name="", arrow_size:float = 1.0, transparent_arrows=True):
    '''
    Erstellt ein 3D-Koordinatensystem mit den Achsen X, Y, Z und optionalen Beschriftungen.

    :param len: Länge der Achsen.
    :param font_scale: Skalierung der Schriftgröße für die Achsenbeschriftungen.
    :param show_labels: Boolescher Wert, der angibt, ob die Achsenbeschriftungen angezeigt werden sollen.
    :param name: Optionaler Name, der in der Mitte des Koordinatensystems angezeigt wird.

    :return: Ein 3D-Objekt (Group), das das Koordinatensystem mit Achsen und optionalen Beschriftungen enthält.
    '''
    line_material_x = three.LineBasicMaterial(color='red')
    line_material_y = three.LineBasicMaterial(color='green')
    line_material_z = three.LineBasicMaterial(color='blue')

    points_x = [[0,0,0], [len,0,0]]
    points_y = [[0,0,0], [0,len,0]]
    points_z = [[0,0,0], [0,0,len]]

    line_geometry_x = three.BufferGeometry(attributes={'position' : three.BufferAttribute(points_x, False)})
    line_geometry_y = three.BufferGeometry(attributes={'position' : three.BufferAttribute(points_y, False)})
    line_geometry_z = three.BufferGeometry(attributes={'position' : three.BufferAttribute(points_z, False)})

    line_x = three.Line(line_geometry_x, line_material_x)
    line_y = three.Line(line_geometry_y, line_material_y)
    line_z = three.Line(line_geometry_z, line_material_z)

    axes_group = three.Group()
    axes_group.add(line_x)
    axes_group.add(line_y)
    axes_group.add(line_z)


    font_offset=0.3

    ttx = TextTexture("X", color='#000000')
    x_label = Sprite(
    material=SpriteMaterial(map=ttx, transparent=True, opacity=0.9, depthWrite=False),
    position=[len+font_offset, 0, 0],
    scale=(font_scale, font_scale, font_scale),
    visible=show_labels
    )

    tty = TextTexture("Y", color='#000000')
    y_label = Sprite(
    material=SpriteMaterial(map=tty, transparent=True, opacity=0.9, depthWrite=False),
    position=[0, len+font_offset, 0],
    scale=(font_scale, font_scale, font_scale),
    visible=show_labels
    )

    ttz = TextTexture("Z", color='#000000')
    z_label = Sprite(
    material=SpriteMaterial(map=ttz, transparent=True, opacity=0.9, depthWrite=False),
    position=[0, 0, len+font_offset],
    scale=(font_scale, font_scale, font_scale),
    visible=show_labels
    )

    axes_group.add([x_label, y_label, z_label])


    cyl_x = create_cylinder([len,0,0], radiusTop=0.1*arrow_size, radiusBottom=0.01*arrow_size, height=0.3*arrow_size, color=[255,0,0], transparent=transparent_arrows)
    rotate(cyl_x, [0,0,90], "XYZ")
    axes_group.add(cyl_x)

    cyl_y = create_cylinder([0,len,0], radiusTop=0.1*arrow_size, radiusBottom=0.01*arrow_size, height=0.3*arrow_size, color=[0,255,0], transparent=transparent_arrows)
    rotate(cyl_y, [180,0,0], "XYZ")
    axes_group.add(cyl_y)

    cyl_z = create_cylinder([0,0,len], radiusTop=0.1*arrow_size, radiusBottom=0.01*arrow_size, height=0.3*arrow_size, color=[0,0,255], transparent=transparent_arrows)
    rotate(cyl_z, [-90,0,0], "XYZ")
    axes_group.add(cyl_z)

    if name!="":
        n = TextTexture(name, color='#000000')
        name_label = Sprite(
        material=SpriteMaterial(map=n, transparent=True, opacity=1, depthWrite=False),
        position=[font_offset, font_offset, font_offset],
        scale=(font_scale, font_scale, font_scale),
        visible=show_labels
        )
        axes_group.add([name_label])

    return axes_group







def create_grid_XY(size, density, pos=[0,0,0], color='#777777'):
    '''
    Erstellt ein 3D-Gitter im XY-Plane mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X und Y Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien im XY-Plane enthält.
    '''
    line_material = three.LineBasicMaterial(color = color)
    line_material.transparent = True
    line_material.opacity = 0.5

    grid_group = three.Group()
    for i in range((int)((-size/2)*(1/density)), (int)((size/2)*(1/density))+1):
        points1 = [[-size/2,i*density,0],[size/2,i*density,0]]
        points2 = [[i*density,-size/2,0],[i*density,size/2,0]]
        # Geometrie für die Linie
        line_geometry1 = three.BufferGeometry(
        attributes={'position': three.BufferAttribute(points1, False)})
        line1 = three.Line(line_geometry1, line_material)
        line_geometry2 = three.BufferGeometry(
        attributes={'position': three.BufferAttribute(points2, False)})
        line2 = three.Line(line_geometry2, line_material)
        grid_group.add(line1)
        grid_group.add(line2)
        grid_group.position = pos
    return grid_group



def create_grid_XZ(size, density, pos=[0,0,0], color='#777777'):
    '''
    Erstellt ein 3D-Gitter im XZ-Plane mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X und Y Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien im XZ-Plane enthält.
    '''
    line_material = three.LineBasicMaterial(color = color)
    line_material.transparent = True
    line_material.opacity = 0.5

    grid_group = three.Group()
    for i in range((int)((-size/2)*(1/density)), (int)((size/2)*(1/density))+1):
        points1 = [[-size/2,0,i*density],[size/2,0,i*density]]
        points2 = [[i*density,0,-size/2],[i*density,0,size/2]]
        # Geometrie für die Linie
        line_geometry1 = three.BufferGeometry(
        attributes={'position': three.BufferAttribute(points1, False)})
        line1 = three.Line(line_geometry1, line_material)
        line_geometry2 = three.BufferGeometry(
        attributes={'position': three.BufferAttribute(points2, False)})
        line2 = three.Line(line_geometry2, line_material)
        grid_group.add(line1)
        grid_group.add(line2)
        grid_group.position = pos
    return grid_group




def apply_transformation_matrix(obj, transform_matrix):
    if isinstance(transform_matrix,(sp.Basic, sp.MatrixBase)):
        transform_matrix.evalf()
    pos = transform_matrix[:3, 3]
    rot = R.from_matrix(transform_matrix[:3, :3]).as_quat()

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    renderable.position = tuple(pos)
    renderable.quaternion = tuple(rot)



def apply_rot_matrix(obj, rot_mat):
    '''
    Wendet eine Rotationsmatrix auf ein Objekt an, indem die Matrix in ein Quaternion umgewandelt wird und auf das bestehende Quaternion des Meshs angewendet wird.
    Das Objekt musse entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, auf das die Rotation angewendet werden soll. Erwartet wird, dass das Mesh ein `quaternion`-Attribut besitzt.
    :param rot_mat: Die Rotationsmatrix, die auf das Mesh angewendet werden soll. Muss eine 3x3 Matrix sein.
    '''
    if isinstance(rot_mat,(sp.Basic, sp.MatrixBase)):
        rot_mat.evalf()
    # Konvertiere Matrix in Quaternion
    r = R.from_matrix(rot_mat)
    q = r.as_quat()  # Reihenfolge: [x, y, z, w]

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()

    # Setze Quaternion (pythreejs erwartet [w, x, y, z])
    renderable.quaternion = quaternion_multiply((q[0], q[1], q[2], q[3]), renderable.quaternion)






def create_colored_quad(pos, width, height, depth,
                        face_colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
                        transparent=True):
    '''
    Erzeugt ein Quader-Mesh mit individuell gefärbten Seiten.

    :param pos: Position des Quaders [x, y, z]
    :param width: Breite des Quaders
    :param height: Höhe des Quaders
    :param depth: Tiefe des Quaders
    :param face_colors: Liste von 6 RGB-Farben, eine pro Seite. Reihenfolge:
                        [right, left, top, bottom, front, back]
    :param transparent: Gibt an, ob das Material transparent ist
    :return: Ein Mesh-Objekt mit verschiedenfarbigen Seiten
    '''
    geometry = three.BoxGeometry(width=width, height=height, depth=depth)

    materials = []
    for rgb in face_colors:
        hex_color = f'#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}'
        mat = three.MeshStandardMaterial(
            color=hex_color,
            metalness=0.5,
            roughness=0.8,
            transparent=transparent,
            opacity=0.5
        )
        materials.append(mat)

    mesh = three.Mesh(geometry=geometry, material=materials)
    mesh.position = tuple(pos)
    return mesh




def create_quad(pos, width, height, depth, color=[0,255,0], transparent=True):
    '''
    Erzeugt ein Quader-Mesh (Box) mit der angegebenen Position, Größe und Farbe.

    :param pos: Die Position des Quaders als Array oder Tuple [x, y, z].
    :param width: Die Breite des Quaders.
    :param height: Die Höhe des Quaders.
    :param depth: Die Tiefe des Quaders.
    :param color: Die Farbe des Quaders als Array [R, G, B], wobei jede Komponente im Bereich 0-255 liegt. Standardmäßig grün [0, 255, 0].
    :param transparent: Ein Boolean-Wert, der angibt, ob das Material transparent sein soll. Standardmäßig `True`.

    :return: Ein Mesh-Objekt, das den Quader darstellt, mit der angegebenen Position, Größe und Farbe.
    '''
    if isinstance(pos,(sp.Basic, sp.MatrixBase)):
        pos.evalf()
    # Erstelle die Geometrie (Breite, Höhe, Tiefe)
    geometry = three.BoxGeometry(width=width, height=height, depth=depth)
    # Material (Farbe & Eigenschaften)
    hex_color = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
    material = three.MeshStandardMaterial(color=hex_color, metalness=0.5, roughness=0.8, transparent=transparent, opacity=0.5)
    # Erstelle das Mesh (Geometrie + Material)
    mesh = three.Mesh(geometry, material)
    mesh.position = (pos[0], pos[1], pos[2])
    return mesh







def create_cylinder(pos, radiusTop=1, radiusBottom=1, height=2, radialSegments=32, color=[255,0,0], transparent=True):
    '''
    Erstellt ein Zylinder-Mesh mit der angegebenen Position, Größe und Farbe.

    :param pos: Die Position des Zylinders als Array oder Tuple [x, y, z].
    :param radiusTop: Der Radius des Zylinders an der Oberseite. Standardwert ist 1.
    :param radiusBottom: Der Radius des Zylinders an der Unterseite. Standardwert ist 1.
    :param height: Die Höhe des Zylinders. Standardwert ist 2.
    :param radialSegments: Die Anzahl der radialen Segmente des Zylinders, die die Auflösung rund um den Zylinder bestimmen. Standardwert ist 32.
    :param color: Die Farbe des Zylinders als Array [R, G, B], wobei jede Komponente im Bereich 0-255 liegt. Standardwert ist [255, 0, 0] (Rot).
    :param transparent: Ein Boolean-Wert, der angibt, ob das Material transparent sein soll. Standardmäßig `True`.

    :return: Ein Mesh-Objekt, das den Zylinder darstellt, mit der angegebenen Position, Größe und Farbe.
    '''
    if isinstance(pos,(sp.Basic, sp.MatrixBase)):
        pos.evalf()
    # Erstelle eine CylinderGeometry
    geometry = CylinderGeometry(
    radiusTop=radiusTop,     # Radius oben
    radiusBottom=radiusBottom,  # Radius unten
    height=height,        # Höhe
    radialSegments=radialSegments  # Auflösung rundherum
    )
    hex_color = f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'
    material = three.MeshStandardMaterial(color=hex_color, metalness=0.5, roughness=0.8, transparent=transparent, opacity=0.5)

    # Mesh aus Geometrie + Material
    cylinder = Mesh(
        geometry=geometry,
        material=material,
        position=pos
    )
    return cylinder







def apply_rot_matrix_animated(obj, rot_mat, speed=100, show_rot_axis=True):
    '''
    Wendet eine Rotationsmatrix auf ein Objekt an und rotiert es animiert mit einer gegebenen Geschwindigkeit.
    Dabei kann optional eine Rotationsachse angezeigt werden.

    Das Objekt musse entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Das Objekt, das rotiert werden soll.
    :param rot_mat: Die Rotationsmatrix, die auf das Mesh angewendet werden soll.
    :param speed: Die Geschwindigkeit der Animation, angegeben als Anzahl der Frames pro Sekunde. Standardwert ist 100.
    :param show_rot_axis: Ein Boolean-Wert, der angibt, ob die Rotationsachse angezeigt werden soll. Standardwert ist `True`.
    '''
    if isinstance(rot_mat,(sp.Basic, sp.MatrixBase)):
        rot_mat.evalf()

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    if show_rot_axis==True:
        a = rot_axis_from_rot_mat(rot_mat)
        material_axis = three.LineBasicMaterial(color='black')
        points_axis = [a*-10, a*10]
        geometry_axis = three.BufferGeometry(attributes={'position' : three.BufferAttribute(points_axis, False)})
        axis = three.Line(geometry_axis, material_axis)

    renderable.add(axis)
    q = R.from_matrix(rot_mat).as_quat() 
    old_quat = renderable.quaternion
    new_quat = quaternion_multiply(renderable.quaternion, (q[0], q[1], q[2], q[3]))
    t = 0
    delta = 0.002
    while(t <= 1):
        n = slerp_quaternion(old_quat, new_quat, t)
        renderable.quaternion = [n[0], n[1], n[2], n[3]]
        t += delta
        time.sleep(1/speed)
    n = slerp_quaternion(old_quat, new_quat, 1)
    renderable.quaternion = [n[0], n[1], n[2], n[3]]

    renderable.remove(axis)






def slerp_quaternion(q1, q2, t):
    '''
    Führt eine Spherical Linear Interpolation (SLERP) zwischen zwei Quaternionen durch.
    Interpoliert die Rotation zwischen q1 und q2 basierend auf dem Interpolationswert t.

    :param q1: Das erste Quaternion, das die Anfangsrotation beschreibt.
    :param q2: Das zweite Quaternion, das die Endrotation beschreibt.
    :param t: Der Interpolationswert, der zwischen 0 und 1 liegen muss. Ein Wert von 0 entspricht der Rotation von q1 und ein Wert von 1 entspricht der Rotation von q2.

    :return: Das interpolierte Quaternion, das die Rotation zwischen q1 und q2 bei dem gegebenen Wert von t beschreibt.
    
    :raises ValueError: Wenn der Interpolationswert t nicht zwischen 0 und 1 liegt.
    '''

    if isinstance(q1,(sp.Basic, sp.MatrixBase)):
        q1.evalf()
    if isinstance(q2,(sp.Basic, sp.MatrixBase)):
        q2.evalf()
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Der Interpolationswert t muss zwischen 0 und 1 liegen. t ist aber {t}")
    

    # Erstelle Rotationsobjekte
    key_times = np.array([0, 1])  # Start (0) und Ende (1)
    key_rots = R.from_quat([q1, q2])  # Quaternionen als Rotation-Objekte

    # SLERP-Interpolation erstellen
    slerp = Slerp(key_times, key_rots)

    # Interpolierte Rotation abrufen
    interpolated_rotation = slerp(t)

    return interpolated_rotation.as_quat()





#vel [x,y,theta]
def line_wheel_driven_robot(dummy, vel, steps):
    '''
    Simuliert die Bewegung eines radgetriebenen Roboters entlang einer Linie und erzeugt dabei Liniensegmente zur Visualisierung des Pfades.

    Bei jedem Schritt wird das Dummy-Objekt gemäß der gegebenen Geschwindigkeit bewegt. In regelmäßigen Abständen (alle 64 Schritte)
    wird ein Liniensegment vom Startpunkt dieses Abschnitts zum aktuellen Punkt erstellt, um die Trajektorie sichtbar zu machen.

    :param dummy: Ein Mesh-Objekt, das die Position und Orientierung des Roboters repräsentiert.
    :param vel: Ein Geschwindigkeitsvektor `[v_x, v_y, ω_z]`, bestehend aus Translation in lokaler X/Y-Richtung und Rotation um Z.
    :param steps: Anzahl der Bewegungs-Iterationen (Zeitschritte).

    :return: Eine `three.Group`, die alle erzeugten Liniensegmente enthält (als Trajektorie).
    '''
    if isinstance(vel,(sp.Basic, sp.MatrixBase)):
        vel.evalf()
    lines = []
    points = []
    for i in range(steps):
        points.append(dummy.position)
        move(dummy, vel)
        if (i%64==0):
            points.append(dummy.position)
            line_material = three.LineBasicMaterial(color='black')
            line_geometry = three.BufferGeometry(attributes={'position' : three.BufferAttribute([points[0], points[len(points)-1]], False)})
            line = three.Line(geometry=line_geometry, material=line_material)
            lines.append(line)
            points = []
    points.append(dummy.position)
    line_material = three.LineBasicMaterial(color='black')
    line_geometry = three.BufferGeometry(attributes={'position' : three.BufferAttribute([points[0], points[len(points)-1]], False)})
    line = three.Line(geometry=line_geometry, material=line_material)
    lines.append(line)
    points = []

    line_group = three.Group()
    line_group.add(lines)
    return line_group




def set_scale(obj, scale):
    '''
    Setzt die Skalierung eines Objekts.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Skalierung angepasst werden soll.
    :param scale: Ein Array, das den Skalierungsfaktor für jede Achse (x, y, z) angibt, z.B. [1, 2, 1].
    '''

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    renderable.scale = scale
    




def set_scale_animated(obj, scale):
    '''
    Setzt die Skalierung eines Objekts animiert, indem es schrittweise die Größe verändert.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Skalierung angepasst werden soll.
    :param scale: Ein Array, das die Ziel-Skalierungswerte für jede Achse (x, y, z) angibt, z.B. [1, 2, 1].
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    old_x = renderable.scale[0]
    old_y = renderable.scale[1]
    old_z = renderable.scale[2]
    t = 0
    delta = 0.02
    while(t<=1):
        current_x = (scale[0]-old_x)*t + old_x
        current_y = (scale[0]-old_y)*t + old_y
        current_z = (scale[0]-old_z)*t + old_z
        renderable.scale = (current_x, current_y, current_z)
        t+=delta
        time.sleep(0.01)


#die angles müssen in der Reihenfolge angegeben werden wie es in der order steht bsp: angles=[y,x,z] order="YXZ"
def rotate_animated(obj, angles, order="ZYZ"):
    '''
    Führt eine animierte lokale Rotation eines Objekts durch. Die Rotation erfolgt achsweise entsprechend der angegebenen Reihenfolge.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    Beispiel: order="YXZ" → angles=[Winkel um Y, Winkel um X, Winkel um Z]

    Während der Animation wird die Rotation in drei Schritten durchgeführt – einer pro Achse – und am Ende exakt auf das Ziel-Quaternion gesetzt,
    um numerische Fehler auszugleichen.

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das rotiert werden soll.
    :param angles: Eine Liste mit Rotationswinkeln (in Grad), die in der Reihenfolge `order` angegeben sind.
    :param order: Die Rotationsreihenfolge (z. B. "ZYZ", "YXZ", etc.)
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()

    if isinstance(angles,(sp.Basic, sp.MatrixBase)):
        angles.evalf()
    q_final = quaternion_multiply(renderable.quaternion, euler_to_quaternion(angles, order))
    time.sleep(0.5)
    delta = 0.5
    if angles[0] < 0:
        delta *= -1
    counter = delta
    while counter <= abs(angles[0]):
        q = euler_to_quaternion([delta, 0, 0], order)
        renderable.quaternion = quaternion_multiply(renderable.quaternion, q)
        counter+=abs(delta)
        time.sleep(0.01)
    time.sleep(0.5)
    delta = 0.5
    if angles[1] < 0:
        delta *= -1
    counter = delta
    while counter <= abs(angles[1]):
        q = euler_to_quaternion([0, delta, 0], order)
        renderable.quaternion = quaternion_multiply(renderable.quaternion, q)
        counter+=abs(delta)
        time.sleep(0.01)
    time.sleep(0.5)
    delta = 0.5
    if angles[2] < 0:
        delta *= -1
    counter = delta
    while counter <= abs(angles[2]):
        q = euler_to_quaternion([0, 0, delta], order)
        renderable.quaternion = quaternion_multiply(renderable.quaternion, q)
        counter+=abs(delta)
        time.sleep(0.01)
    renderable.quaternion = q_final
    time.sleep(0.5)
    



#die angles müssen in der Reihenfolge angegeben werden wie es in der order steht bsp: angles=[y,x,z] order="YXZ"
def rotate_global_animated(obj, angles, order="ZYZ"):
    '''
    Führt eine animierte globale Rotation eines Objekts durch. Die Drehung erfolgt achsweise gemäß der angegebenen Rotationsreihenfolge (z. B. "ZYZ").
    Die Winkel in `angles` müssen in der **Reihenfolge der `order`-Zeichen** angegeben werden.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    Beispiel: Bei order="YXZ" → angles=[Winkel um Y, Winkel um X, Winkel um Z]

    Die Funktion führt die Rotation in drei separaten animierten Phasen durch – jeweils eine für jede Achse in `order`.

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das rotiert werden soll.
    :param angles: Eine Liste von Rotationswinkeln (in Grad), entsprechend der Reihenfolge in `order`.
    :param order: Die Rotationsreihenfolge als String, z. B. "ZYZ", "YXZ", etc.
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()

    if isinstance(angles,(sp.Basic, sp.MatrixBase)):
        angles.evalf()
    time.sleep(0.5)
    delta = 0.5
    if angles[0] < 0:
        delta *= -1
    counter = delta
    while counter < abs(angles[0]):
        q = euler_to_quaternion([delta, 0, 0], order)
        renderable.quaternion = quaternion_multiply(q, renderable.quaternion)
        counter+=abs(delta)
        time.sleep(0.01)
    time.sleep(0.5)
    delta = 0.5
    if angles[1] < 0:
        delta *= -1
    counter = delta
    while counter < abs(angles[1]):
        q = euler_to_quaternion([0, delta, 0], order)
        renderable.quaternion = quaternion_multiply(q, renderable.quaternion)
        counter+=abs(delta)
        time.sleep(0.01)
    time.sleep(0.5)
    delta = 0.5
    if angles[2] < 0:
        delta *= -1
    counter = delta
    while counter < abs(angles[2]):
        q = euler_to_quaternion([0, 0, delta], order)
        renderable.quaternion = quaternion_multiply(q, renderable.quaternion)
        counter+=abs(delta)
        time.sleep(0.01)
    time.sleep(0.5)


#vel=[x,y,theta]
def move(robot, vel, steps=1):
    '''
    Bewegt ein Roboterobjekt in mehreren Schritten entsprechend der gegebenen Geschwindigkeit und Rotation.

    Die Funktion kombiniert Translation und Rotation:
    - Zuerst wird eine Rotation um die Z-Achse angewendet, basierend auf dem dritten Element des Geschwindigkeitsvektors `vel[2]`.
    - Anschließend wird eine Translation basierend auf der aktuellen Ausrichtung (Z-Rotation) des Roboters ausgeführt.
    - Die Bewegung wird für eine angegebene Anzahl von Schritten (`steps`) wiederholt.
    - Nach jeweils 4 Schritten erfolgt eine kurze Pause zur visuellen Glättung.

    :param robot: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das bewegt werden soll.
    :param vel: Ein Geschwindigkeitsvektor `[v_x, v_y, ω_z]`, wobei `v_x` und `v_y` die Translation in der lokalen X- und Y-Richtung und `ω_z` die Rotation um die Z-Achse ist (in Radiant).
    :param steps: Die Anzahl der Schritte, die die Bewegung ausführen soll (Standard: 1).

    :return: Die finale Position des Roboters nach der Bewegung.
    '''
    renderable = robot
    if (hasattr(robot, 'get_renderable')):
        renderable = robot.get_renderable()

    if isinstance(vel,(sp.Basic, sp.MatrixBase)):
        vel.evalf()
    for i in range(steps):
        rot_mat_z = np.array([
        [np.cos(vel[2]), -np.sin(vel[2]), 0],
        [np.sin(vel[2]),  np.cos(vel[2]), 0],
        [0,             0,             1]
        ])

        apply_rot_matrix(renderable, rot_mat_z)
        x = renderable.quaternion[0]
        y = renderable.quaternion[1]
        z = renderable.quaternion[2]
        w = renderable.quaternion[3]

        z_angle = quaternion_to_euler(x,y,z,w,"XYZ")[2]
        cos_z = np.cos(np.radians(z_angle))
        sin_z = np.sin(np.radians(z_angle))
        translate(renderable, [cos_z*vel[0] + sin_z*vel[1], sin_z*vel[0] + cos_z*vel[1], 0])
        if (i%4==0 and i!=0):
            time.sleep(0.01)
    return renderable.position








def rotate_global(obj, angles, order="ZYZ"):
    '''
    Führt eine globale Rotation eines Objekts durch, basierend auf den übergebenen Eulerwinkeln und einer Rotationsreihenfolge.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das rotiert werden soll.
    :param angles: Die Eulerwinkel in Grad, die die Rotation definieren. Die Reihenfolge muss dem angegebenen "order"-Parameter entsprechen.
    :param order: Die Rotationsreihenfolge als String (z.B. "ZYZ", "XYZ", etc.). Standardmäßig "ZYZ".
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    renderable.quaternion = quaternion_multiply(euler_to_quaternion(angles, order[::-1]), renderable.quaternion)






def rotate(obj, angles, order="ZYZ"):
    '''
    Führt eine Rotation eines Objekts basierend auf den übergebenen Eulerwinkeln und einer Rotationsreihenfolge durch.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das rotiert werden soll.
    :param angles: Die Eulerwinkel in Grad, die die Rotation definieren. Die Reihenfolge muss dem angegebenen "order"-Parameter entsprechen.
    :param order: Die Rotationsreihenfolge als String (z.B. "ZYZ", "XYZ", etc.). Standardmäßig "ZYZ".

    :return: Keine Rückgabe. Das Mesh wird direkt rotiert.
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    q = euler_to_quaternion(angles, order)
    renderable.quaternion = quaternion_multiply(renderable.quaternion, q)






def set_rotation(obj, angles, order="ZYZ"):
    '''
    Setzt die Rotation eines Objekts auf die übergebenen Eulerwinkel und die Rotationsreihenfolge.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Rotation gesetzt werden soll.
    :param angles: Die Eulerwinkel in Grad, die die gewünschte Rotation definieren. Die Reihenfolge muss dem angegebenen "order"-Parameter entsprechen.
    :param order: Die Rotationsreihenfolge als String (z.B. "ZYZ", "XYZ", etc.). Standardmäßig "ZYZ".
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    q = euler_to_quaternion(angles, order=order)
    renderable.quaternion = tuple(q)






def set_rotation_global(obj, angles, order="ZYZ"):
    '''
    Setzt die globale Rotation eines Objekts auf die übergebenen Eulerwinkel und die Rotationsreihenfolge. 

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen globale Rotation gesetzt werden soll.
    :param angles: Die Eulerwinkel in Grad, die die gewünschte Rotation definieren. Die Reihenfolge muss dem angegebenen "order"-Parameter entsprechen.
    :param order: Die Rotationsreihenfolge als String (z.B. "ZYZ", "XYZ", etc.). Standardmäßig "ZYZ".
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    set_rotation(renderable, angles[::-1], order[::-1])






def translate(obj, vec):
    '''
    Verschiebt ein Objekt um einen gegebenen Vektor in den drei Raumachsen.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, das verschoben werden soll.
    :param vec: Der Verschiebungsvektor als Array oder Liste [x, y, z], der die Verschiebung in den jeweiligen Raumachsen angibt.
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    if isinstance(vec,(sp.Basic, sp.MatrixBase)):
        vec.evalf()
    renderable.position = (renderable.position[0]+vec[0], renderable.position[1]+vec[1], renderable.position[2]+vec[2])






def set_translation(obj, vec):
    '''
    Setzt die Position eines Objekts auf die angegebenen Koordinaten.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Position gesetzt werden soll.
    :param vec: Der Ziel-Vektor als Array oder Liste [x, y, z], der die neue Position des Meshs im Raum angibt.
    '''
    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()
    if isinstance(vec,(sp.Basic, sp.MatrixBase)):
        vec.evalf()
    renderable.position = tuple(vec)






def set_translation_animated(obj, vec, speed=50.0):
    '''
    Bewegt die Position eines Objekts animiert von der aktuellen Position zu einer angegebenen Zielposition.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Position animiert geändert werden soll.
    :param vec: Der Ziel-Vektor als Array oder Liste [x, y, z], zu dem die Position des Meshs bewegt werden soll.
    :param speed: Die Geschwindigkeit der Animation. Ein höherer Wert bedeutet eine schnellere Bewegung.
    '''

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()

    if isinstance(vec,(sp.Basic, sp.MatrixBase)):
        vec.evalf()
    t = 0
    delta = 0.01
    old_x = renderable.position[0]
    old_y = renderable.position[1]
    old_z = renderable.position[2]
    while(t<=1):
        current_x = ((vec[0]-old_x)*t + old_x)
        current_y = ((vec[1]-old_y)*t + old_y)
        current_z = ((vec[2]-old_z)*t + old_z)
        renderable.position = (current_x, current_y, current_z)
        t+=delta
        time.sleep(1.0/speed)
    current_x = ((vec[0]-old_x)*1 + old_x)
    current_y = ((vec[1]-old_y)*1 + old_y)
    current_z = ((vec[2]-old_z)*1 + old_z)
    renderable.position = (current_x, current_y, current_z)
    time.sleep(1.0/speed)





def translate_animated(obj, vec, speed=50.0):
    '''
    Bewegt die Position eines Objekts animiert um einen angegebenen Vektor von der aktuellen Position.

    Das Objekt muss entweder:
    - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
    - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z.B. Mesh, Group, Line, etc.).

    :param obj: Ein renderbares Objekt oder ein Wrapper mit `get_renderable()`, dessen Position animiert geändert werden soll.
    :param vec: Der Verschiebungs-Vektor als Array oder Liste [dx, dy, dz], um den die Position des Meshs verändert werden soll.
    :param speed: Die Geschwindigkeit der Animation. Ein höherer Wert bedeutet eine schnellere Bewegung.
    '''

    renderable = obj
    if (hasattr(obj, 'get_renderable')):
        renderable = obj.get_renderable()

    if isinstance(vec,(sp.Basic, sp.MatrixBase)):
        vec.evalf()
    t = 0
    delta = 0.01
    old_x = renderable.position[0]
    old_y = renderable.position[1]
    old_z = renderable.position[2]
    while(t<=1):
        current_x = ((vec[0])*t + old_x)
        current_y = ((vec[1])*t + old_y)
        current_z = ((vec[2])*t + old_z)
        renderable.position = (current_x, current_y, current_z)
        t+=delta
        time.sleep(1.0/speed)
    current_x = ((vec[0])*1 + old_x)
    current_y = ((vec[1])*1 + old_y)
    current_z = ((vec[2])*1 + old_z)
    renderable.position = (current_x, current_y, current_z)
    time.sleep(1.0/speed)
        



def create_differential_robot():
    '''
    Erzeugt einen Cylinderförmigen Roboter mit Differentialantrieb.
    Der Roboter hat zwei Räder

    :return: Ein 3D-Objekt (Mesh) das den Differentialroboter darstellt.
    '''
    robot_group = three.Group()

    wheel_height = 0.2
    wheel_radius = 0.4
    robot_radius = 1
    chassis = create_cylinder([0, 0, wheel_radius], robot_radius, robot_radius, 0.5, 32, [0,255,0], True)

    w0 = create_cylinder([0, 0, 1+wheel_height/2], wheel_radius, wheel_radius, wheel_height, 32, [255,0,0], True)
    rotate(w0, [90,0,0], "XYZ")
    w1 = create_cylinder([0, 0, -1-wheel_height/2], wheel_radius, wheel_radius, wheel_height, 32, [255,0,0], True)
    rotate(w1, [90,0,0], "XYZ")

    w0_axis = create_axes(2)
    w1_axis = create_axes(2)

    #w0.add([w0, w0_axis])
    #w1.add([w1, w1_axis])

    chassis.add([w0, w1])
    rotate(chassis, [-90,0,0], "XYZ")
    robot_axis = create_axes(2)
    robot_group.add([chassis, robot_axis])
    return robot_group




class Environment:
    '''
    Eine 3D-Umgebung, die eine Szene mit Kamera, Lichtquellen, Achsen, Gittern und Widgets für die Interaktivität erstellt.

    Diese Klasse erstellt eine 3D-Umgebung mit einer Vielzahl von Features, darunter eine Kamera, Lichtquellen, Achsen- und Gitterdarstellung sowie Steuerungen zur Anpassung von Objekten in der Szene (z.B. Rotation, Skalierung, Translation).

    :param width: Die Breite der Ansicht in Pixeln (Standard: 700).
    :param height: Die Höhe der Ansicht in Pixeln (Standard: 500).
    :param frame: Ein 3D-Achsenobjekt, das als Referenzrahmen in der Szene hinzugefügt wird (Standard: create_axes(8, name="B")).
    :param grid: Ein Gitterobjekt, das in der Szene angezeigt wird (Standard: create_grid_XY(14, 0.5)).
    :param up: Die Richtung der "Oben"-Achse, die die Orientierung der Kamera bestimmt (Standard: [0, 0, 1]).

    Diese Klasse enthält Methoden, um:
    - die Sichtbarkeit von Gitter und Achsen zu steuern,
    - interaktive Widgets für Objekte zu erstellen (Translation, Rotation, Skalierung),
    - Objekte der Szene hinzuzufügen,
    - globale oder lokale Transformationen auf Objekte anzuwenden.

    Weitere Features:
    - Die Umgebung kann mit einer interaktiven Steuerung für Kamera und Objekte angezeigt werden.
    - Widgets für die Manipulation von Objekten (Translation, Rotation, Skalierung) können zur Szene hinzugefügt werden.
    '''
    def __init__(self, width=700, height=500, frame=None, grid=None, up=[0,0,1], widgets_on_bottom=False):
        '''
        Initialisiert eine neue 3D-Umgebung.

        :param width: Die Breite der Ansicht.
        :param height: Die Höhe der Ansicht.
        :param frame: Ein 3D-Achsenobjekt, das in der Szene hinzugefügt wird.
        :param grid: Ein Gitterobjekt, das in der Szene angezeigt wird.
        :param up: Die Richtung der "Oben"-Achse für die Kamera.
        '''
        if frame is None:
            frame = create_axes(8, name="B")
        if grid is None:
            grid = create_grid_XY(14,0.5)
        self.frame = frame
        self.grid = grid
        self.scene = Scene()
        self.scene.background = "#DDDDDD"
        self.camera = PerspectiveCamera(position=[8, 8, 8],aspect=width/height, fov=50)
        self.camera.up = up
        self.frame = frame
        self.grid = grid
        self.light = PointLight(color='white', intensity=1.5, position=[5, 5, 5])
        self.scene.add([self.camera, self.light, self.frame, self.grid, AmbientLight(intensity=0.5)])
        self.children = []
        self.widgets_on_bottom = widgets_on_bottom
        # Renderer mit Orbit-Steuerung
        self.renderer = Renderer(camera=self.camera, scene=self.scene, controls=[OrbitControls(controlling=self.camera)], width=width, height=height, background_color="#87CEEB", background_opacity=1.0, antialias=True, precision='highp')
        self.frame_widgets = True
        self.widgets = []
        self.gizmo_controls = []
        self.inspectors = []

        self.info_container:widgets.VBox = widgets.VBox()
        


        
        


    def toggle_grid(self, change):
        '''
        Schaltet die Sichtbarkeit des Gitters um.

        :param change: Das Ereignis, das diese Funktion auslöst (wird nicht genutzt).
        '''
        self.grid.visible = not self.grid.visible




    def toggle_axes(self, change):
        '''
        Schaltet die Sichtbarkeit der Achsen um.

        :param change: Das Ereignis, das diese Funktion auslöst (wird nicht genutzt).
        '''
        self.frame.visible = not self.frame.visible




    def _ipython_display_(self):
        '''
        Zeigt die Umgebung mit Renderer und interaktiven Widgets an, wenn sie in einem Jupyter-Notebook verwendet wird.
        '''
        layout2 = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            flex='none' 
        )

        mainbox = VBox([self.renderer], layout = layout2)
        if self.frame_widgets:
            checkbox_grid = Checkbox(value=True, description='Show Grid')
            checkbox_axes = Checkbox(value=True, description='Show Axes')
            #interactive_control_scale = widgets.interactive(update_cube_scale, x=self.x_scale_slider, y=self.y_scale_slider, z=self.z_scale_slider)
            checkbox_grid.observe(self.toggle_grid, names='value')
            checkbox_axes.observe(self.toggle_axes, names='value')
            frame_widget_box = HBox([checkbox_grid, checkbox_axes])
            mainbox.children = mainbox.children + (frame_widget_box,)


        layout1 = widgets.Layout(
            #border='1px solid gray',
            padding='5px',
            height=f'{self.renderer.height}px',
            #width ='1600px',
            overflow_y='auto',
            overflow_x = "auto",
            flex="none"
        )

        

        widget_box = VBox(children=[], layout = layout1)
        for w in self.widgets:
            widget_box.children = widget_box.children + (w,)
            w.layout.overvlow="hidden"

        #lol_box = HBox(children = [widget_box], layout=layout2)
        if self.widgets_on_bottom:
            display(VBox(children = [mainbox, widget_box]))
        else:
            display(HBox(children = [mainbox, widget_box]))
        display(self.info_container)


        
    def set_frame_widgets(self, bool):
        '''
        Aktiviert oder deaktiviert die Anzeige von Frame-Widgets.

        :param bool: Wenn True, werden die Widgets angezeigt, andernfalls ausgeblendet.
        '''
        self.frame_widgets = bool


    def add(self, objects):
        """
        Fügt ein oder mehrere Objekte zur Environment hinzu.

        Die Objekte müssen entweder:
        - ein Attribut oder eine Methode `get_renderable()` besitzen, das ein pythreejs-kompatibles Objekt zurückgibt,
        - oder selbst direkt ein pythreejs-kompatibles Objekt sein (z. B. Mesh, Group, Line, etc.).

        Hinweis: Es erfolgt keine explizite Typprüfung. Es wird davon ausgegangen, dass übergebene Objekte entweder
        direkt von pythreejs unterstützt werden oder über `get_renderable()` ein entsprechendes Objekt liefern.

        :param objekts: Ein einzelnes Objekt oder eine Liste von Objekten, die zur Environment hinzugefügt werden sollen.
        :raises AttributeError: Wenn `get_renderable()` aufgerufen wird, aber nicht vorhanden ist.
        :raises TraitError / TypeError: Wenn ein nicht unterstützter Objekttyp der Szene hinzugefügt wird.
        """
        if hasattr(objects, "set_environment") : objects.set_environment(self)
        if isinstance(objects, Object3D):
            self.scene.add(objects)
            self.children.append(objects)
        elif hasattr(objects, "get_renderable"):
            self.scene.add(objects.get_renderable())
            self.children.append(objects)
        elif isinstance(objects, Iterable):
            for obj in objects:
                if hasattr(obj, "set_environment") : obj.set_environment(self)
                if isinstance(obj, Object3D):
                    self.scene.add(obj)
                    self.children.append(obj)
                elif hasattr(obj, "get_renderable"):
                    self.scene.add(obj.get_renderable())
                    self.children.append(obj)




    def add_widget(self, widget):
        '''
        Fügt ein Widget zur Umgebung hinzu. Dabei kann es sich auch um ein Buendel von Widgets in einer HBox oder einer VBox handeln.

        :param widget: Das Widget, das der Umgebung hinzugefügt werden soll.
        '''
        self.widgets.append(widget)
    
    def add_info(self, info_text:str):
        test_button:widgets.Button = widgets.Button(
            description='',
            tooltip='',
            icon='times',
            layout=widgets.Layout(width='32px')
        )
        
        test_label = widgets.HTML(value=f'<span style="font-size:14px; color:red;">{info_text}</span>')
        info = HBox(children=[test_label, test_button])
        self.info_container.children = list(self.info_container.children) + [info]
        def on_click(button):
            self.remove_info(info)
        test_button.on_click(on_click)
        return info

    def remove_info(self, info):
        self.info_container.children = tuple(w for w in self.info_container.children if w != info)
    


    def add_gizmo_controls(
            self, obj, translation=True, rotation=True, scale=False, name="",
            max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
            min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
            widgets_vertical:Bool = False,
            continuous_update=True,
            callback: Callable[[], None] = None):
        '''
        Fügt ein Gizmo-Steuerelement zur Manipulation eines Objekts in der Umgebung hinzu (Translation, Rotation, Skalierung).

        :param obj: Das Objekt, das manipuliert werden soll.
        :param translation: Wenn True, werden Schieberegler für die Translation angezeigt.
        :param rotation: Wenn True, werden Schieberegler für die Rotation angezeigt.
        :param scale: Wenn True, werden Schieberegler für die Skalierung angezeigt.
        '''
        controls = self.Gizmo_Controls(obj, translation, rotation, scale, name,
                                    max_trans_x, max_trans_y, max_trans_z,
                                    min_trans_x, min_trans_y, min_trans_z,
                                    widgets_vertical, continuous_update, callback)
        self.gizmo_controls.append(controls)
        self.add_widget(controls.widget)
        return controls


    def add_inspector(self, obj):
        if hasattr(obj, "add_inspector"):
            self.inspectors.append(obj.add_inspector(self))




#----------------------------------------GIZMO_CONTROLS---------------------------------------------------------


    class Gizmo_Controls:
        layout1_horizontal = widgets.Layout(
                #border='1px solid gray',
                padding='5px',
                #width='100',
                height='125px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
                )
            
        layout1_vertical = widgets.Layout(
                #border='1px solid gray',
                padding='5px',
                #width='100',
                #height='125px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
            )
        
        layout2_horizontal = widgets.Layout(
                border='1px solid gray',
                padding='5px',
                #width='100',
                height='200px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
            )
        
        layout2_vertical = widgets.Layout(
                border='1px solid gray',
                padding='5px',
                #width='100',
                max_height='500px',
                overflow='hidden',  # Scrollen deaktivieren
                flex='none'
            )


        def __init__(
                self, obj, translation=True, rotation=True, scale=False, name="",
                max_trans_x:float = 1, max_trans_y:float = 1, max_trans_z:float = 1,
                min_trans_x:float = -1, min_trans_y:float = -1, min_trans_z:float = -1,
                widgets_vertical:Bool = False,
                continuous_update=True,
                callback: Callable[[], None] = None):
            '''
            Erstellt ein Gizmo-Steuerelement zur Manipulation eines Objekts (Translation, Rotation, Skalierung).

            :param obj: Das Objekt, das manipuliert werden soll.
            :param translation: Wenn True, werden Schieberegler für die Translation angezeigt.
            :param rotation: Wenn True, werden Schieberegler für die Rotation angezeigt.
            :param scale: Wenn True, werden Schieberegler für die Skalierung angezeigt.
            '''
            
            self.callback = callback
            self.obj = obj
            self.content = []
            if name != "":
                self.content.append(Label(name))
            self.obj_renderable = obj
            if hasattr(obj, "get_renderable"):
                self.obj_renderable = obj.get_renderable()

            
            if widgets_vertical:
                layout1 = self.__class__.layout1_vertical
                layout2 = self.__class__.layout2_vertical
            else :    
                layout1 = self.__class__.layout1_horizontal
                layout2 = self.__class__.layout2_horizontal

            if translation:
                self.x_trans_slider = FloatSlider(min=min_trans_x, max = max_trans_x, step=0.001, description="Translation X", continuous_update=continuous_update)
                self.y_trans_slider = FloatSlider(min=min_trans_y, max = max_trans_y, step=0.001, description="Translation Y", continuous_update=continuous_update)
                self.z_trans_slider = FloatSlider(min=min_trans_z, max = max_trans_z, step=0.001, description="Translation Z", continuous_update=continuous_update)
                self.x_trans_slider.value = self.obj_renderable.position[0]
                self.y_trans_slider.value = self.obj_renderable.position[1]
                self.z_trans_slider.value = self.obj_renderable.position[2]

                

                trans_box = VBox(children=[self.x_trans_slider, self.y_trans_slider, self.z_trans_slider], layout=layout1)
                
                self.content.append(trans_box)

                self.x_trans_slider.observe(self._on_trans_slider, names='value')
                self.y_trans_slider.observe(self._on_trans_slider, names="value")
                self.z_trans_slider.observe(self._on_trans_slider, names="value")

            if rotation:
                self.x_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate X', continuous_update=continuous_update)
                self.y_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Y', continuous_update=continuous_update)
                self.z_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Z', continuous_update=continuous_update)
                rot = R.from_quat(list(self.obj_renderable.quaternion))
                euler = rot.as_euler("XYZ", degrees=True) 
                self.x_rot_slider.value = euler[0]
                self.y_rot_slider.value = euler[1]
                self.z_rot_slider.value = euler[2]
                rot_box = VBox(children=[self.x_rot_slider, self.y_rot_slider, self.z_rot_slider], layout=layout1)
                self.content.append(rot_box)


                self.x_rot_slider.observe(self._on_rot_slider, names="value")
                self.y_rot_slider.observe(self._on_rot_slider, names="value")
                self.z_rot_slider.observe(self._on_rot_slider, names="value")

                

            if scale:
                self.x_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale X", value=1, continuous_update=continuous_update)
                self.y_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Y", value=1, continuous_update=continuous_update)
                self.z_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Z", value=1, continuous_update=continuous_update)
                scale_box = VBox(children=[self.x_scale_slider, self.y_scale_slider, self.z_scale_slider], layout=layout1)
                self.content.append(scale_box)

                def _on_scale_slider(change):
                    set_scale(self.obj_renderable, [self.x_scale_slider.value, self.y_scale_slider.value, self.z_scale_slider.value])
                    if callback is not None : callback()

                self.x_scale_slider.observe(_on_scale_slider, names="value")
                self.y_scale_slider.observe(_on_scale_slider, names="value")
                self.z_scale_slider.observe(_on_scale_slider, names="value")



            if rotation:
                #ZYX ist Roll Nick Gier wie in der Vorlesung, ZYZ ist Euler wie in der Vorlesung
                self.rotation_order_dropdown = Dropdown(
                    options=['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', "ZYZ", "XYX", "XZX", "YXY", "YZY", "ZXZ"],
                    value='XYZ',
                    description='Rotation Order:',
                )

                self.rotation_order_dropdown.observe(self._on_rotation_order_change, names='value')  
                


            self.local_space_check_box = widgets.Checkbox(value=False, description="Lokale Transformation", layout=widgets.Layout(width='350px', height='30px'))
            self.local_space_check_box.observe(self._on_local_space_check_box, names='value')

            if widgets_vertical:
                box = VBox(children = self.content, layout = layout1)
            else:
                box = HBox(children = self.content, layout = layout1)
            if rotation :
                main_box = VBox(children = [box, self.rotation_order_dropdown, self.local_space_check_box], layout=layout2)
            else :
                main_box = VBox(children = [box, self.local_space_check_box], layout=layout2)
            self.widget = main_box
            
        





        def set_translation_silently(self, translation:list):
            self.x_trans_slider.unobserve(self._on_trans_slider, names="value")
            self.y_trans_slider.unobserve(self._on_trans_slider, names="value")
            self.z_trans_slider.unobserve(self._on_trans_slider, names="value")
            self.x_trans_slider.value = translation[0]
            self.y_trans_slider.value = translation[1]
            self.z_trans_slider.value = translation[2]
            self.x_trans_slider.observe(self._on_trans_slider, names="value")
            self.y_trans_slider.observe(self._on_trans_slider, names="value")
            self.z_trans_slider.observe(self._on_trans_slider, names="value")
            



        def set_rotation_silently(self, rot_xyz):
            self.x_rot_slider.unobserve(self._on_rot_slider, names="value")
            self.y_rot_slider.unobserve(self._on_rot_slider, names="value")
            self.z_rot_slider.unobserve(self._on_rot_slider, names="value")
            self.x_rot_slider.value = rot_xyz[0]
            self.y_rot_slider.value = rot_xyz[1]
            self.z_rot_slider.value = rot_xyz[2]
            self.x_rot_slider.observe(self._on_rot_slider, names="value")
            self.y_rot_slider.observe(self._on_rot_slider, names="value")
            self.z_rot_slider.observe(self._on_rot_slider, names="value")

        

        def _on_trans_slider(self, change):#noch fehler drin
            delta = change["new"] - change["old"]
            v = None
            if change['owner'] is self.x_trans_slider: v = np.array([delta, 0, 0])
            if change['owner'] is self.y_trans_slider: v = np.array([0, delta, 0])
            if change['owner'] is self.z_trans_slider: v = np.array([0, 0, delta])
            rot_mat = R.from_quat(list(self.obj_renderable.quaternion)).as_matrix()
            if self.local_space_check_box.value:
                final_v = rot_mat @ v
            else : final_v = v
            self.obj_renderable.position = tuple(np.array([self.obj_renderable.position[0], self.obj_renderable.position[1], self.obj_renderable.position[2]]) + np.array(final_v))
            if self.callback is not None : self.callback()


        

        def _on_rot_slider(self, change):
            o = self.rotation_order_dropdown.value
            if (o == "zyz" or o == "ZYZ" or
                o == "xyx" or o == "XYX" or
                o == "xzx" or o == "XZX" or
                o == "yxy" or o == "YXY" or
                o == "yzy" or o == "YZY" or
                o == "zxz" or o == "ZXZ"):
                if self.local_space_check_box.value:
                    set_rotation(self.obj_renderable, [self.x_rot_slider.value, self.y_rot_slider.value, self.z_rot_slider.value], self.rotation_order_dropdown.value)
                else:
                    set_rotation_global(self.obj_renderable, [self.x_rot_slider.value, self.y_rot_slider.value, self.z_rot_slider.value], self.rotation_order_dropdown.value)
            else:
                angles = order_angles([self.x_rot_slider.value, self.y_rot_slider.value, self.z_rot_slider.value], "XYZ", self.rotation_order_dropdown.value)
                if self.local_space_check_box.value:
                    set_rotation(self.obj_renderable, angles, self.rotation_order_dropdown.value)
                else:
                    set_rotation_global(self.obj_renderable, angles, self.rotation_order_dropdown.value)
            if self.callback is not None : self.callback()



        def _on_local_space_check_box(self, change):
            self.x_rot_slider.unobserve(self._on_rot_slider, names="value")
            self.y_rot_slider.unobserve(self._on_rot_slider, names="value")
            self.z_rot_slider.unobserve(self._on_rot_slider, names="value")
            if self.local_space_check_box.value:
                euler = quaternion_to_euler(self.obj.quaternion[0], self.obj.quaternion[1], self.obj.quaternion[2], self.obj.quaternion[3], self.rotation_order_dropdown.value)
                euler = order_angles(euler, self.rotation_order_dropdown.value, "XYZ")
                self.x_rot_slider.value = euler[0]
                self.y_rot_slider.value = euler[1]
                self.z_rot_slider.value = euler[2]
            else:
                euler = quaternion_to_euler(self.obj.quaternion[0], self.obj.quaternion[1], self.obj.quaternion[2], self.obj.quaternion[3], self.rotation_order_dropdown.value[::-1])
                euler = order_angles(euler, self.rotation_order_dropdown.value[::-1], "XYZ")
                self.x_rot_slider.value = euler[0]
                self.y_rot_slider.value = euler[1]
                self.z_rot_slider.value = euler[2]
            self.x_rot_slider.observe(self._on_rot_slider, names="value")
            self.y_rot_slider.observe(self._on_rot_slider, names="value")
            self.z_rot_slider.observe(self._on_rot_slider, names="value")






        def _on_rotation_order_change(self, change):
            o = self.rotation_order_dropdown.value
            if (o=="ZYZ" or o=="zyz"):
                self.x_rot_slider.description="Rotate Z"
                self.y_rot_slider.description="Rotate Y"
                self.z_rot_slider.description="Rotate Z"
            elif (o=="XYX" or o=="xyx"):
                self.x_rot_slider.description="Rotate X"
                self.y_rot_slider.description="Rotate Y"
                self.z_rot_slider.description="Rotate X"
            elif (o=="XZX" or o=="xzx"):
                self.x_rot_slider.description="Rotate X"
                self.y_rot_slider.description="Rotate Z"
                self.z_rot_slider.description="Rotate X"
            elif (o=="YXY" or o=="yxy"):
                self.x_rot_slider.description="Rotate Y"
                self.y_rot_slider.description="Rotate X"
                self.z_rot_slider.description="Rotate Y"
            elif (o=="YZY" or o=="yzy"):
                self.x_rot_slider.description="Rotate Y"
                self.y_rot_slider.description="Rotate Z"
                self.z_rot_slider.description="Rotate Y"
            elif (o=="ZXZ" or o=="zxz"):
                self.x_rot_slider.description="Rotate Z"
                self.y_rot_slider.description="Rotate X"
                self.z_rot_slider.description="Rotate Z"
            else:
                self.x_rot_slider.description="Rotate X"
                self.y_rot_slider.description="Rotate Y"
                self.z_rot_slider.description="Rotate Z"
            self._on_rot_slider(None)


                
            



    


    