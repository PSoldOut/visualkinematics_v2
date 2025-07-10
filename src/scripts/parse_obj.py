import numpy as np
import pywavefront
import visualkinematics_v2.util as util
import os
from multiprocessing import Pool, cpu_count

def parse(filepath): 
    if os.path.isfile(filepath):
        obj_model = pywavefront.Wavefront(
            filepath,
            collect_faces=True,
            create_materials=False,
            parse=True
            )  # Die 'model.obj' Datei laden
        vertices = np.array(obj_model.vertices)
        #indices = np.array(obj_model.meshes[0].faces).flatten()

        indices = []
        for name, mesh in obj_model.meshes.items():
            indices.extend(mesh.faces)
        indices = np.array(indices, dtype=np.uint32).flatten()
        normals = util.compute_normals(vertices, indices)
        #normals = np.array(obj_model.parser.normals)

        filename_with_extension = os.path.basename(filepath)
        filename_without_extension, _ = os.path.splitext(filename_with_extension)  # Trennt Dateinamen und Endung
        output_filepath = os.path.join("../assets/npz/", f"{filename_without_extension}.npz")
        np.savez(output_filepath, vertices=vertices, indices=indices, normals=normals)
        print("...parsed " + filename_with_extension + "...")

if __name__ == "__main__":
    in_path = "../assets/obj/"
    out_path = "../assets/npz/"
    print("parsing from obj to npz...")
    os.makedirs(in_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    filepaths = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.endswith(".obj")]

    # Verwende alle verfügbaren CPU-Kerne
    with Pool(cpu_count()) as pool:
        pool.map(parse, filepaths)