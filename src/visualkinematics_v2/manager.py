from __future__ import annotations
import numpy as np
import pythreejs as three
from ipywidgets import *
from pythreejs import *
from stl import mesh
import os
from xml.etree import ElementTree as ET
from pathlib import Path
import xml.etree.ElementTree as ET
import xacrodoc as xd
from typing import List
import trimesh
from lxml.etree import XMLSyntaxError
from collada.common import DaeMalformedError
import subprocess
import pywavefront
from visualkinematics_v2 import util
from appdirs import user_config_dir
import toml
from pathlib import Path
import visualkinematics_v2


#Datenhaltung (Model)


APP_NAME = "visualkinematics_v2"
CONFIG_FILENAME = "config.toml"

DEFAULT_CONFIG = {
    "base_paths": [
        str(Path(visualkinematics_v2.__file__).parent / "_package_data"),
        str(Path.home() / "Documents" / "visualkinematics_v2" / "assets" / "custom-package-sets"),
        str(Path.home() / "Documents" / "visualkinematics_v2" / "assets" / "ros-package-sets")
    ],

    "teach_path" : str(Path.home() / "Documents" / "visualkinematics_v2" / "res" / "teach"),
    "npz_path" :  str(Path.home() / "Documents" / "visualkinematics_v2" / "assets" / "npz"),
    "obj_path" :  str(Path.home() / "Documents" / "visualkinematics_v2" / "assets" / "obj")
}



def load_config():
    config_dir = user_config_dir(APP_NAME)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, CONFIG_FILENAME)

    # Wenn die Datei nicht existiert, eine Default-Konfig schreiben
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            toml.dump(DEFAULT_CONFIG, f)

    with open(config_path, "r") as f:
        return toml.load(f)

config = load_config()



#project_path = Path(__file__).resolve().parent.parent.parent 
#os.chdir(project_path)
teach_path = config["teach_path"]
npz_path = config["npz_path"]
obj_path = config["obj_path"]
base_paths = config["base_paths"]

fast_load : bool = True

package_root = Path(visualkinematics_v2.__file__).parent
#print(package_root)









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
                elif child.tag == "limit":
                    joint_info["limit"] = child.attrib


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
                clean_filename = "unbekannt"
                if filename is not None:
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





















def to_npz(filepath_in, filepath_out):
    # Zielordner erstellen, falls nicht vorhanden
    target_dir = Path(filepath_out).parent
    os.makedirs(target_dir, exist_ok=True)

    try:
        mesh = trimesh.load(filepath_in, force='mesh')
    except (XMLSyntaxError, DaeMalformedError) as e:
        print(f"[WARN] Fehler beim Laden von {filepath_in}: {e}")
        print("[INFO] Versuche, Datei zu bereinigen ...")
        if filepath_in.endswith(".dae"):
            mesh = _clean_dae(filepath_in)

    if mesh.is_empty:
        raise ValueError(f"Fehler beim Laden der Mesh-Datei (leer oder ungültig): {filepath_in}")

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    indices = faces.flatten()
    normals = util.compute_normals(vertices, indices)

    np.savez(filepath_out, vertices=vertices, indices=indices, normals=normals)
    print(f"...parsed {filepath_out}")






def _clean_dae(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Nur bis zum Ende des </COLLADA>-Tags behalten
    end_index = content.find("</COLLADA>")
    if end_index != -1:
        clean_content = content[:end_index + len("</COLLADA>")]

        # Temporäre Datei erzeugen
        temp_path = filepath.replace(".dae", "_cleaned.dae")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(clean_content)

        print(f"[INFO] Neuversuch mit bereinigter Datei: {temp_path}")
        return trimesh.load(temp_path, force='mesh')
    else:
        raise RuntimeError("Kein </COLLADA>-Tag gefunden. Datei ist möglicherweise komplett beschädigt.")
    




def load_mesh_auto_compatibility(filepath, color="lightgray", opacity=1.0, robot_name:str = "robot"):
    if fast_load:
        npz_filepath = f"{npz_path}/{robot_name}/{Path(filepath).stem}.npz"
        if not os.path.isfile(npz_filepath):
            to_npz(filepath, npz_filepath)  
        return load_mesh_from_npz(npz_filepath, color)


    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mesh-Datei nicht gefunden: {filepath}")
    try:
        mesh = trimesh.load(filepath, force='mesh')
    except (XMLSyntaxError, DaeMalformedError) as e:
        print(f"[WARN] Fehler beim Laden von {filepath}: {e}")
        print("[INFO] Versuche, Datei zu bereinigen ...")
        if filepath.endswith(".dae"):
            mesh = _clean_dae(filepath)
        else:
            raise e

    if mesh.is_empty:
        raise ValueError(f"Fehler beim Laden der Mesh-Datei (leer oder ungültig): {filepath}")

    # Vertices & Faces extrahieren
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # BufferGeometry bauen
    geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(vertices, normalized=False),
            'index': BufferAttribute(faces.flatten(), normalized=False),
        }
    )
    geometry.exec_three_obj_method('computeVertexNormals')

    material = MeshStandardMaterial(color=color, opacity=opacity, transparent=True)
    mesh_obj = Mesh(geometry=geometry, material=material)
    return mesh_obj












def load_mesh_from_npz(filepath, color="lightgray"):
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

    obj_material = MeshStandardMaterial(color=color)
    obj_mesh = three.Mesh(obj_geometry, material=obj_material)

    return obj_mesh