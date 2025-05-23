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
import manager
from collections.abc import Iterable
import typing









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




def compute_normals(vertices, indices):
    '''
    Generiert die Normalen fuer alle Dreiecke die sich aus vertices und indices ergeben.

    :param vertices: Die Punkte im Raum aus denen die Dreiecke bestehen, für die die Normalen berechnet werden als Array.
    :param indices: Die Indexe zum Verbinden der Punke zu Dreiecken, für welche dann die Normalen berechnet werden als Array.

    :return: Ein Array, welches die Normalen enthaelt.
    '''
    # Initialisiere Array für Normalen
    normals = np.zeros_like(vertices)

    # Für jedes Dreieck die Flächennormale berechnen
    for i in range(0, len(indices), 3):
        idx1, idx2, idx3 = indices[i], indices[i+1], indices[i+2]
        v1, v2, v3 = vertices[idx1], vertices[idx2], vertices[idx3]

        # Berechne zwei Kanten des Dreiecks
        edge1 = v2 - v1
        edge2 = v3 - v1

        # Kreuzprodukt für die Flächennormale
        face_normal = np.cross(edge1, edge2)

        # Normalisiere die Flächennormale
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Addiere die Flächennormale zu den Vertex-Normalen
        normals[idx1] += face_normal
        normals[idx2] += face_normal
        normals[idx3] += face_normal

    # Normalisiere die Vertex-Normalen
    normals = np.array([n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n for n in normals])
    return normals





def order_angles(x, y, z, order):
    '''
    Ordnet die uebergebenen Eulerwinkel nach order und gibt diese als Array zurueck.

    :param x: Eulerwinkel um x.
    :param y: Eulerwinkel um y.
    :param z: Eulerwinkel um z.

    :return: Die geordneten Eulerwinkel als Array.
    '''
    if order == "ZYZ" or order == "zyz":
        return [z,y,z]
    elif order == "XYX" or order == "xyx":
        return [x,y,x]
    elif order == "XZX" or order == "xzx":
        return [x,z,x]
    elif order == "YXY" or order == "yxy":
        return [y,x,y]
    elif order == "YZY" or order == "yzy":
        return [y,z,y]
    elif order == "ZXZ" or order == "zxz":
        return [z,x,z]
    elif order == "XYZ" or order == "xyz":
        return [x,y,z]
    elif order == "XZY" or order == "xzy":
        return [x,z,y] 
    elif order == "YZX" or order == "yzx":
        return [y,z,x]
    elif order == "YXZ" or order == "yxz":
        return [y,x,z]
    elif order == "ZXY" or order == "zxy":
        return [z,x,y]
    elif order == "ZYX" or order == "zyx":
        return [z,y,x]




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








def create_axes(len, font_scale=0.4, show_labels=True, name=""):
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


    cyl_x = create_cylinder([len,0,0], radiusTop=0.1, radiusBottom=0.01, height=0.3, color=[255,0,0])
    rotate(cyl_x, [0,0,90], "XYZ")
    axes_group.add(cyl_x)

    cyl_y = create_cylinder([0,len,0], radiusTop=0.1, radiusBottom=0.01, height=0.3, color=[0,255,0])
    rotate(cyl_y, [180,0,0], "XYZ")
    axes_group.add(cyl_y)

    cyl_z = create_cylinder([0,0,len], radiusTop=0.1, radiusBottom=0.01, height=0.3, color=[0,0,255])
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







def create_grid_XY(size, density):
    '''
    Erstellt ein 3D-Gitter im XY-Plane mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X und Y Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien im XY-Plane enthält.
    '''
    line_material = three.LineBasicMaterial(color='#777777')
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
    return grid_group



def create_grid_XZ(size, density):
    '''
    Erstellt ein 3D-Gitter im XZ-Plane mit der angegebenen Größe und Dichte.

    :param size: Die Größe des Gitters (die Ausdehnung in X und Y Richtung).
    :param density: Die Dichte des Gitters, die angibt, wie viele Linien innerhalb des Gitters erstellt werden.

    :return: Ein 3D-Objekt (Group), das das Gitter mit Linien im XZ-Plane enthält.
    '''
    line_material = three.LineBasicMaterial(color='#777777')
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
    return grid_group




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
        raise ValueError("Der Interpolationswert t muss zwischen 0 und 1 liegen.")
    
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
    renderable.quaternion = [q[0], q[1], q[2], q[3]]






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
    renderable.position = vec






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
    def __init__(self, width=700, height=500, frame=None, grid=None, up=[0,0,1]):
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
        # Renderer mit Orbit-Steuerung
        self.renderer = Renderer(camera=self.camera, scene=self.scene, controls=[OrbitControls(controlling=self.camera)], width=width, height=height, background_color="#87CEEB", background_opacity=1.0, antialias=True, precision='highp')
        self.frame_widgets = True
        self.widgets = []



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
        display(self.renderer)
        if self.frame_widgets:
            checkbox_grid = Checkbox(value=True, description='Show Grid')
            checkbox_axes = Checkbox(value=True, description='Show Axes')
            #interactive_control_scale = widgets.interactive(update_cube_scale, x=x_scale_slider, y=y_scale_slider, z=z_scale_slider)
            checkbox_grid.observe(self.toggle_grid, names='value')
            checkbox_axes.observe(self.toggle_axes, names='value')
            display(HBox([checkbox_grid, checkbox_axes]))

        for w in self.widgets:
            display(w)
        

        
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
        if isinstance(objects, Object3D):
            self.scene.add(objects)
            self.children.append(objects)
        elif hasattr(objects, "get_renderable"):
            self.scene.add(objects.get_renderable())
            self.children.append(objects)
        elif isinstance(objects, Iterable):
            for obj in objects:
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
    

    
    def add_gizmo_controls(self, obj, translation=True, rotation=True, scale=True):
        '''
        Fügt ein Gizmo-Steuerelement zur Manipulation eines Objekts in der Umgebung hinzu (Translation, Rotation, Skalierung).

        :param obj: Das Objekt, das manipuliert werden soll.
        :param translation: Wenn True, werden Schieberegler für die Translation angezeigt.
        :param rotation: Wenn True, werden Schieberegler für die Rotation angezeigt.
        :param scale: Wenn True, werden Schieberegler für die Skalierung angezeigt.
        '''
        renderable = obj
        if hasattr(obj, "get_renderable"):
            renderable = obj.get_renderable()

        # Schieberegler
        x_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate X')
        y_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Y')
        z_rot_slider = FloatSlider(min=-180, max=180, step=0.1, description='Rotate Z')

        x_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale X", value=1)
        y_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Y", value=1)
        z_scale_slider = FloatSlider(min=0, max=5, step=0.001, description="Scale Z", value=1)

        x_trans_slider = FloatSlider(min=0, max = 10, step=0.001, description="Translation X")
        y_trans_slider = FloatSlider(min=0, max = 10, step=0.001, description="Translation Y")
        z_trans_slider = FloatSlider(min=0, max = 10, step=0.001, description="Translation Z")


        #ZYX ist Roll Nick Gier wie in der Vorlesung, ZYZ ist Euler wie in der Vorlesung
        rotation_order_dropdown = Dropdown(
            options=['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', "ZYZ", "XYX", "XZX", "YXY", "YZY", "ZXZ"],
            value='ZYX',
            description='Rotation Order:',
        )

        

        trans_box = VBox([x_trans_slider, y_trans_slider, z_trans_slider])
        rot_box = VBox([x_rot_slider, y_rot_slider, z_rot_slider])
        scale_box = VBox([x_scale_slider, y_scale_slider, z_scale_slider])

        box = HBox([trans_box, rot_box, scale_box])


        self.add_widget(box)
        self.add_widget(rotation_order_dropdown)



        def _on_trans_slider(change):#noch fehler drin
            set_translation(renderable, [x_trans_slider.value, y_trans_slider.value, z_trans_slider.value])

        def _on_rot_slider(change):
            o = rotation_order_dropdown.value
            if (o == "zyz" or o == "ZYZ" or
                o == "xyx" or o == "XYX" or
                o == "xzx" or o == "XZX" or
                o == "yxy" or o == "YXY" or
                o == "yzy" or o == "YZY" or
                o == "zxz" or o == "ZXZ"):
                set_rotation(renderable, [x_rot_slider.value, y_rot_slider.value, z_rot_slider.value], rotation_order_dropdown.value)
            else:
                angles = order_angles(x_rot_slider.value, y_rot_slider.value, z_rot_slider.value, rotation_order_dropdown.value)
                set_rotation(renderable, angles, rotation_order_dropdown.value)
    

        def _on_scale_slider(change):
            set_scale(renderable, [x_scale_slider.value, y_scale_slider.value, z_scale_slider.value])


        def _on_rotation_order_change(change):
            o = rotation_order_dropdown.value
            if (o=="ZYZ" or o=="zyz"):
                x_rot_slider.description="Rotate Z"
                y_rot_slider.description="Rotate Y"
                z_rot_slider.description="Rotate Z"
            elif (o=="XYX" or o=="xyx"):
                x_rot_slider.description="Rotate X"
                y_rot_slider.description="Rotate Y"
                z_rot_slider.description="Rotate X"
            elif (o=="XZX" or o=="xzx"):
                x_rot_slider.description="Rotate X"
                y_rot_slider.description="Rotate Z"
                z_rot_slider.description="Rotate X"
            elif (o=="YXY" or o=="yxy"):
                x_rot_slider.description="Rotate Y"
                y_rot_slider.description="Rotate X"
                z_rot_slider.description="Rotate Y"
            elif (o=="YZY" or o=="yzy"):
                x_rot_slider.description="Rotate Y"
                y_rot_slider.description="Rotate Z"
                z_rot_slider.description="Rotate Y"
            elif (o=="ZXZ" or o=="zxz"):
                x_rot_slider.description="Rotate Z"
                y_rot_slider.description="Rotate X"
                z_rot_slider.description="Rotate Z"
            else:
                x_rot_slider.description="Rotate X"
                y_rot_slider.description="Rotate Y"
                z_rot_slider.description="Rotate Z"

            _on_rot_slider(None)
       
        
        x_trans_slider.observe(_on_trans_slider, names='value')
        y_trans_slider.observe(_on_trans_slider, names="value")
        z_trans_slider.observe(_on_trans_slider, names="value")
        x_rot_slider.observe(_on_rot_slider, names="value")
        y_rot_slider.observe(_on_rot_slider, names="value")
        z_rot_slider.observe(_on_rot_slider, names="value")
        x_scale_slider.observe(_on_scale_slider, names="value")
        y_scale_slider.observe(_on_scale_slider, names="value")
        z_scale_slider.observe(_on_scale_slider, names="value")

        rotation_order_dropdown.observe(_on_rotation_order_change, names='value')    




class Kinematic_Chain_Element:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None
        self.renderable = None

        
    def add(self, object):
        renderable = object
        if hasattr(object, "get_renderable"):
            renderable = object.get_renderable()
        self.renderable.add(renderable)
        self.children.append(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = self


    def remove(self, object):
        self.children.remove(object)
        if hasattr(object, "get_renderable"):
            self.renderable.remove(object.get_renderable())
        elif isinstance(object, Object3D):
            self.renderable.remove(object)
        if isinstance(object, Kinematic_Chain_Element):
            object.parent = None



class Joint(Kinematic_Chain_Element):
    def __init__(self, name, axis, position=[0,0,0], rotation=[0,0,0]):
        super().__init__(name)
        self.renderable = three.Group()
        self.axis = axis
        self.mimicers : List[List] = []
        self.set_position(position)
        self.set_rotation(rotation)

    def add_mimicer(self, mimicer_and_multiplier : List):
        self.mimicers.append(mimicer_and_multiplier)
        


    def get_renderable(self):
        return self.renderable
    
    def set_position(self, vec):
        set_translation(self.renderable, vec)

    def get_position(self):
        return self.renderable.position

    def set_rotation(self, vec):
        set_rotation(self.renderable, vec, "ZYX")

    def get_rotation(self):
        """
        Gibt die aktuelle Rotation als Euler-Winkel in Grad im 'ZYX'-Format zurück.

        :return: Liste von drei Winkeln [z, y, x] in Grad.
        """
        q = self.renderable.quaternion  # Quaternion: [x, y, z, w]
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        euler_deg = r.as_euler("ZYX", degrees=True)
        return euler_deg.tolist()





class Link(Kinematic_Chain_Element):
    def __init__(self, name, mesh):
        super().__init__(name)

        self.renderable = mesh

    def get_renderable(self):
        return self.renderable
    

        



class Manipulator:
    def __init__(self, name):
        self.name=name
        self.mesh = None
        self.links = []
        self.joints = []
        xacro_filepath = manager.find_xacro_filepath_by_robot_name(name)
        urdf = manager.xacro_to_urdf_string(xacro_filepath)
        #print(urdf)
        parsed = manager.parse_urdf(urdf)
        parsed = json.dumps(parsed, indent=4)
        print(parsed)
        parsed = json.loads(parsed)

        
        for link_element in parsed["links"]:
            meshi = None
            if len(link_element["visual"]) > 0:
                mesh_path = link_element["visual"][0]["geometry"]["filename"]
                if link_element["visual"][0]["material"] is not None:
                    material_name : str = link_element["visual"][0]["material"]["name"]
                    rgba = None
                    if material_name == "":
                        rgba : str
                    else:
                        meshi = manager.load_mesh_auto(mesh_path, link_element["visual"][0]["material"]["name"])
                else:
                    meshi = manager.load_mesh_auto(mesh_path)
            else:
                meshi = three.Group()
            l = Link(link_element["name"], meshi)
            self.links.append(l)
                
                
        for i in range(len(parsed["joints"])):
            joint_element = parsed["joints"][i]
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
            joint : Joint = Joint(joint_element["name"], joint_axis, pos, angles)
            joint.add(joint_child)
            joint_parent.add(joint) 
            if joint_element["mimic"] is not None:
                multiplier : float = float(joint_element["mimic"]["multiplier"])
                gets_mimiced : Joint = self.get_joint_by_name(joint_element["mimic"]["joint"])
                gets_mimiced.add_mimicer([joint, multiplier])
            self.joints.append(joint)
            print("mimicer:")
            for m in joint.mimicers:
                print(m[0].name, " name")
                print(m[1], " mult")

        self.mesh = self.links[0].get_renderable()

    
    def get_renderable(self):
        return self.mesh
    
    def get_link_by_name(self, name : str):
        for l in self.links:
            if l.name == name:
                return l
        return None

    def get_joint_by_name(self, name : str):
        for j in self.joints:
            if j.name == name:
                return j
        return None
    
    def print_links(self):
        for link in self.links:
            print(link.name)

    def print_joints(self):
        for joint in self.joints:
            print(joint.name)

    def print_kinematic_chain(self):
        current = self.links[0]
        print(current.name)
        while(len(current.children) > 0):
            current = current.children[0]
            print(current.name)








def apply_joint_angle(joint: Joint, axis, angle_rad):
    """
    Wendet eine Rotation um eine gegebene Achse auf das Joint-Objekt an.

    :param joint: Joint-Instanz
    :param axis: 3D-Achse als Liste [x, y, z]
    :param angle_rad: Winkel in Radiant
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)  # Normalisieren
    r = R.from_rotvec(axis * angle_rad)  # Rotationsvektor → Quaternion
    

    quat = r.as_quat()  # [x, y, z, w] Reihenfolge!
    joint.get_renderable().quaternion = (quat[0], quat[1], quat[2], quat[3])
    for m in joint.mimicers:
        r = R.from_rotvec(axis * (angle_rad*m[1]))
        q = r.as_quat()
        m[0].get_renderable().quaternion = (q[0], q[1], q[2], q[3])



def apply_joint_rotation(joint : Joint, axis, angle_rad):
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

    # Ausgangsrotation (aus URDF z. B.)
    base_rot = R.from_euler("ZYX", joint.get_rotation(), degrees=True)

    # Neue Rotation um die Achse
    axis_rot = R.from_rotvec(axis * angle_rad)

    # Reihenfolge ist entscheidend: zuerst lokale Basisrotation, dann Gelenkwinkel
    final_rot = axis_rot * base_rot

    # Quaternion setzen
    q = final_rot.as_quat()
    joint.get_renderable().quaternion = (q[0], q[1], q[2], q[3])

    #mimicers
    for m in joint.mimicers:
        base_rot = R.from_euler("ZYX", m[0].get_rotation(), degrees=True)
        axis_rot = R.from_rotvec(axis * (angle_rad*m[1]))
        final_rot = axis_rot * base_rot
        q = final_rot.as_quat()
        m[0].get_renderable().quaternion = (q[0], q[1], q[2], q[3])
        

def animation(joint : Joint, axis, angle_rad):
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
        QuaternionKeyframeTrack(name='.quaternion', times=[0,4], values=[q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]]), 
    ]
    clip : AnimationClip = AnimationClip(tracks=tracks, duration=4)
    action : AnimationAction = AnimationAction(AnimationMixer(joint.get_renderable()), clip, joint.get_renderable())
    action.play()
    


    