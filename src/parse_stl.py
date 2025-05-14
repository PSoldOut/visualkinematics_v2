import numpy as np
from stl import mesh
import os
from multiprocessing import Pool, cpu_count
import util
import sys

def stl_to_npz(filepath_info):
    filepath, in_path, out_path = filepath_info
    robot_name = os.path.basename(os.path.dirname(filepath))
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Zielordner für diesen Roboter
    target_dir = os.path.join(out_path, robot_name)
    os.makedirs(target_dir, exist_ok=True)

    # STL laden
    stl_mesh = mesh.Mesh.from_file(filepath)
    triangles = stl_mesh.vectors
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse_indices.reshape(-1, 3)

    normals = util.compute_normals(unique_vertices, indices.flatten())

    output_filepath = os.path.join(target_dir, f"{filename}.npz")
    np.savez(output_filepath, vertices=unique_vertices, indices= faces.flatten().astype(np.uint32), normals=normals)
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
