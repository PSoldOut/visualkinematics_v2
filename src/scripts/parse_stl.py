import numpy as np
from stl import mesh
import os
from multiprocessing import Pool, cpu_count
import visualkinematics_v2.util as util
import sys



def compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
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




def stl_to_npz(filepath_info):
    filepath, in_path, out_path = filepath_info
    robot_name = os.path.basename(os.path.dirname(filepath))
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Zielordner für diesen Roboter
    target_dir = os.path.join(out_path, robot_name)
    os.makedirs(target_dir, exist_ok=True)

    # STL laden
    stl_mesh = mesh.Mesh.from_file(filepath)
    vertices = np.array(stl_mesh.vectors).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    
    indices = faces.flatten().astype(np.uint32)
    normals = compute_vertex_normals(vertices, indices)
    

    output_filepath = os.path.join(target_dir, f"{filename}.npz")
    np.savez(output_filepath, vertices=vertices, indices=indices, normals=normals)
    print(f"...parsed {robot_name}/{filename}.stl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bitte einen Eingabeordner angeben: python convert.py <in_path>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = "../assets/npz/"
    os.makedirs(out_path, exist_ok=True)

    # Alle .stl-Dateien rekursiv suchen
    all_stl_files = []
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file.endswith(".stl"):
                full_path = os.path.join(root, file)
                all_stl_files.append((full_path, in_path, out_path))

    print(f"Gefundene STL-Dateien: {len(all_stl_files)}")

    with Pool(cpu_count()) as pool:
        pool.map(stl_to_npz, all_stl_files)
