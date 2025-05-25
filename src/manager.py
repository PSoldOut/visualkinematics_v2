import numpy as np
import sympy as sp
import pythreejs as three
from ipywidgets import *
from IPython.display import display
from pythreejs import *
from pythreejs import SpriteMaterial, Sprite
import time
from scipy.spatial.transform import Rotation as R, Slerp
from stl import mesh
import os
from lxml import etree
import xacro
from io import StringIO
from xml.etree import ElementTree as ET
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile
from xacrodoc import XacroDoc
import xacrodoc as xd
from typing import List
import trimesh



base_paths = [
    "../assets/ros-package-sets",
    "../assets/custom-package-sets"
]







def find_xacro_filepath_by_robot_name(robot_name: str, base_paths: List[str] = base_paths) -> str:
    """
    Durchsucht rekursiv alle Pfade in base_paths nach einer .xacro-Datei mit dem Namen <robot_name>.xacro,
    ignoriert dabei aber <robot_name>_macro.xacro.

    Args:
        base_paths: Liste von Basisverzeichnissen, in denen gesucht werden soll.
        robot_name: Name des Roboters (z.B. "irb1600_10_145").

    Returns:
        Vollständiger Pfad zur gefundenen .xacro-Datei.

    Raises:
        FileNotFoundError: Wenn keine passende Datei gefunden wurde.
    """
    expected_filename = f"{robot_name}.xacro"

    for base_path in base_paths:
        for root, _, files in os.walk(base_path):
            for file in files:
                if file == expected_filename:
                    return os.path.join(root, file)

    raise FileNotFoundError(
        f"Keine passende Datei {expected_filename} in den übergebenen Pfaden gefunden: {base_paths}"
    )






def find_ros_packages(base_paths: list[str]) -> dict:
    package_map = {}
    for base in base_paths:
        base_path = Path(base).expanduser().resolve()
        for pkg_file in base_path.rglob("package.xml"):
            try:
                tree = ET.parse(pkg_file)
                root = tree.getroot()
                name_tag = root.find("name")
                if name_tag is not None:
                    package_name = name_tag.text.strip()
                    package_map[package_name] = str(pkg_file.parent)
            except Exception as e:
                print(f"Fehler beim Parsen von {pkg_file}: {e}")
    return package_map




#vulnerabel weil abhängig vom packer xacrodoc. support nicht gesichert. vielleicht in zukunft besser eine ros installation mit ins framework zu bringen
def xacro_to_urdf_string(xacro_file_path: str, package_paths = base_paths, mappings: dict = {}) -> str:
    xd.packages.look_in(package_paths)
    return xd.XacroDoc.from_file(xacro_file_path, walk_up=False, subargs=mappings).to_urdf_string()






def parse_urdf(urdf_str: str) -> dict:
    root = ET.fromstring(urdf_str)

    links = []
    joints = []

    for elem in root:
        if elem.tag == "link":
            link_info = {
                "name": elem.attrib.get("name"),
                "visual": [],
                "collision": [],
                "inertial": None
            }

            for child in elem:
                if child.tag == "visual":
                    vis = parse_geometry_block(child)
                    if vis:
                        link_info["visual"].append(vis)

                elif child.tag == "collision":
                    col = parse_geometry_block(child)
                    if col:
                        link_info["collision"].append(col)

                elif child.tag == "inertial":
                    # Optional: Trägheitsdaten
                    link_info["inertial"] = {
                        k.tag: k.attrib for k in child
                    }

            links.append(link_info)

        elif elem.tag == "joint":
            joint_info = {
                "name": elem.attrib.get("name"),
                "type": elem.attrib.get("type"),
                "parent": None,
                "child": None,
                "origin": None,
                "axis": None,
                "mimic": None
            }

            for child in elem:
                if child.tag == "parent":
                    joint_info["parent"] = child.attrib.get("link")
                elif child.tag == "child":
                    joint_info["child"] = child.attrib.get("link")
                elif child.tag == "origin":
                    joint_info["origin"] = child.attrib
                elif child.tag == "axis":
                    joint_info["axis"] = child.attrib
                elif child.tag == "mimic":
                    joint_info["mimic"] = child.attrib



            joints.append(joint_info)

    return {
        "links": links,
        "joints": joints
    }

def parse_geometry_block(tag):
    """Parst <visual> oder <collision> und gibt Pfad, origin, scale etc. zurück"""
    info = {
        "origin": None,
        "geometry": None,
        "material": None
    }

    for child in tag:
        if child.tag == "origin":
            info["origin"] = child.attrib

        elif child.tag == "geometry":
            for geo in child:
                filename = geo.attrib.get("filename")
                clean_filename = filename.replace("file://", "")
                if geo.tag == "mesh":
                    info["geometry"] = {
                        "type": "mesh",
                        "filename": clean_filename,
                        "scale": geo.attrib.get("scale", "1 1 1")
                    }
                else:
                    # Box, cylinder, sphere, etc.
                    info["geometry"] = {
                        "type": geo.tag,
                        "params": geo.attrib
                    }

        elif child.tag == "material":
            name = child.attrib.get("name")
            rgba = None
            for subChild in child:
                if subChild.tag == "color":
                    rgba = subChild.attrib.get("rgba")
            info["material"] = {
                "name" : name,
                "rgba" : rgba
            }
            

    return info if info["geometry"] else None







def load_mesh_auto(filepath, color="lightgray", opacity=1.0):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh-Datei nicht gefunden: {filepath}")
    
    mesh = trimesh.load(filepath, force='mesh')
    
    if mesh.is_empty:
        raise ValueError(f"Fehler beim Laden der Mesh-Datei (leer oder ungültig): {filepath}")
    

    # Vertices & Faces extrahieren
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    # BufferGeometry bauen
    geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(vertices, normalized=False),
            'index': BufferAttribute(faces.flatten(), normalized=False),  # Dreiecksindizes
        }
        
    )
    geometry.exec_three_obj_method('computeVertexNormals')
    # Mesh erstellen
    material = MeshStandardMaterial(color=color, opacity=opacity, transparent=True)
    mesh_obj = Mesh(geometry=geometry, material=material)
    return mesh_obj





def load_mesh_from_stl(filepath):
    # STL laden
    your_mesh = mesh.Mesh.from_file(filepath)

    # Vertices & Faces extrahieren
    vertices = np.array(your_mesh.vectors).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    # BufferGeometry bauen
    geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(vertices, normalized=False),
            'index': three.BufferAttribute(faces.flatten().astype(np.uint32), normalized=False),  # Indices der Dreiecke
        },
        
    )
    geometry.exec_three_obj_method('computeVertexNormals')

    # Mesh erstellen
    material = MeshStandardMaterial(color='orange')
    mesh_obj = Mesh(geometry=geometry, material=material)
    return mesh_obj




def load_mesh_from_npz(filepath):
    data = np.load(filepath)
    vertices = data["vertices"]
    indices = data["indices"]
    normals = data["normals"]

    obj_geometry = three.BufferGeometry(
        attributes={
            'position': three.BufferAttribute(vertices, normalized=False),  # Positionsdaten
            'index': three.BufferAttribute(indices, normalized=False),  # Indices der Dreiecke
            'normal': three.BufferAttribute(normals, normalized=False),  # Hinzufügen der Normalen
        }
    )
    #obj_geometry.exec_three_obj_method('computeVertexNormals')

    obj_material = MeshStandardMaterial(color='orange')
    #mat = list(obj_model.materials.values())[0]
    #obj_material = MeshStandardMaterial(
    #    color = to_hex(mat.diffuse),
    #    metalness=0.2,
    #    roughness=1 - mat.shininess / 100 if mat.shininess else 0.5
    #)
    obj_mesh = three.Mesh(obj_geometry, material=obj_material)

    return obj_mesh