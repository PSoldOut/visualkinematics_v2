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
    Erstellt ein 3D-Koordinatensystem mit den Achsen X, Y, Z, optionalen Beschriftungen und Pfeilspitzen.

    :param len: Länge der Achsen.
    :param font_scale: Skalierung der Schriftgröße für die Achsenbeschriftungen.
    :param show_labels: Boolescher Wert, der angibt, ob die Achsenbeschriftungen angezeigt werden sollen.
    :param name: Optionaler Name, der in der Mitte des Koordinatensystems angezeigt wird.
    :param arrow_size: Skalierungsfaktor für die Größe der Pfeilspitzen an den Achsen.
    :param transparent_arrows: Boolescher Wert, der angibt, ob die Pfeilspitzen transparent dargestellt werden sollen.

    :return: Ein 3D-Objekt (Group), das das Koordinatensystem mit Achsen, Pfeilspitzen, optionalen Beschriftungen und optionalem Namen enthält.
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
    Erstellt ein 3D-Gitter in der XY-Ebene mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X- und Y-Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.
    :param pos: Die Position des Gitters im Raum als [x, y, z]-Koordinaten.
    :param color: Die Linienfarbe des Gitters (Hex-String, z. B. '#777777').

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien in der XY-Ebene enthält.
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
    Erstellt ein 3D-Gitter in der XZ-Ebene mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X- und Z-Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.
    :param pos: Die Position des Gitters im Raum als [x, y, z]-Koordinaten.
    :param color: Die Linienfarbe des Gitters (Hex-String, z. B. '#777777').

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien in der XY-Ebene enthält.
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
    '''
    Wendet eine Transformationsmatrix auf ein 3D-Objekt an. Die Matrix bestimmt
    die neue Position und Orientierung (Rotation) des Objekts im Raum.

    :param obj: Das Zielobjekt, auf das die Transformation angewendet wird. Kann ein direktes
                3D-Objekt oder ein Objekt mit einer `get_renderable`-Methode sein.
    :param transform_matrix: Eine 4x4-Transformationsmatrix (z. B. aus SymPy oder NumPy),
                             die Translation und Rotation des Objekts definiert.

    :return: None. Das übergebene Objekt wird in-place transformiert.
    '''
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




def create_box(pos, width, height, depth, color=[0,255,0], transparent=True):
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





                
            



    


    