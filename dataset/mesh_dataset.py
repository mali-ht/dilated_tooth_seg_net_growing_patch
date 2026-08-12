import json
import os
import pickle
from os.path import join
from pathlib import Path
from typing import Optional

import numpy as np
import pyfqmr
import torch
import trimesh
from scipy import spatial
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.preprocessing import MoveToOriginTransform
from utils.mesh_io import filter_files
from utils.teeth_numbering import colors_to_label, fdi_to_label


def process_mesh(mesh: trimesh, labels: torch.tensor = None):
    mesh_faces = torch.from_numpy(mesh.faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces]).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces]).float()
    if labels is None:
        labels = torch.from_numpy(colors_to_label(mesh.visual.face_colors.copy())).long()
    return mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels


class Teeth3DSDataset(Dataset):

    def __init__(self, root: str, raw_folder: str = 'raw', processed_folder: str = 'processed_torch',
                 in_memory: bool = False, verbose: bool = True, pre_transform=None, post_transform=None,
                 force_process=False, train_test_split=1, is_train=True, target_density: float = 5.0,
                 max_target_count: Optional[int] = 16000):
        """
        :param target_density: Working density in faces/mm^2 that each downsampled arch is targeted at
            (target face count = round(surface_area_mm^2 * target_density)). Default 5.0 matches the
            live scanner's working density (W).
        :param max_target_count: Upper bound on the density-based target face count, since real arches
            vary a lot in surface area. Keeps individual arches from growing large enough to make the
            O(N^2) full-arch attention prohibitively expensive. Pass None for no cap (true target_density
            everywhere, regardless of arch size) - use a dedicated processed_folder for this, since it is
            not compatible with a capped cache built at the same target_density.
        """
        self.root = root
        self.processed_folder = processed_folder
        self.raw_folder = raw_folder
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.in_memory = in_memory
        self.in_memory_data = []
        self.verbose = verbose
        self.file_names = []
        self.train_test_split = train_test_split
        self.target_density = target_density
        self.max_target_count = max_target_count
        self._set_file_index(is_train)
        self.move_to_origin = MoveToOriginTransform()
        Path(join(self.root, self.processed_folder)).mkdir(parents=True, exist_ok=True)
        Path(join(self.root, self.raw_folder)).mkdir(parents=True, exist_ok=True)
        self._check_cache_config()
        if not self._is_processed() or force_process:
            self._process()
        self.processed_file_names = filter_files(join(self.root, self.processed_folder), 'pt')
        if self.in_memory:
            self._load_in_memory()
        
    def _set_file_index(self, is_train: bool):
        if self.train_test_split == 1:
            split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',
                                                                                         'testing_upper.txt']
        elif self.train_test_split == 2:
            split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
                else ['private-testing-set.txt']
        else:
            raise ValueError(f'train_test_split should be 1 or 2. not {self.train_test_split}')
        for f in split_files:
            with open(f'data/3dteethseg/raw/{f}') as file:
                for l in file:
                    l = f'data_{l.rstrip()}.pt'
                    if os.path.isfile(join(self.root, self.processed_folder, l)):
                        self.file_names.append(l)


    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _loop(self, data):
        if self.verbose:
            return tqdm(data)
        return data
    
    def _donwscale_mesh(self, mesh, labels):
        target_count = round(mesh.area * self.target_density)
        if self.max_target_count is not None:
            target_count = min(target_count, self.max_target_count)
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(target_count=target_count, aggressiveness=3, preserve_border=True, verbose=0,
                                      max_iterations=2000)
        new_positions, new_face, _ = mesh_simplifier.getMesh()
        mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
        vertices = mesh_simple.vertices
        faces = mesh_simple.faces
        if faces.shape[0] < target_count:
            fs_diff = target_count - faces.shape[0]
            faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
        elif faces.shape[0] > target_count:
            mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
            samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, target_count)
            mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
            faces = mesh_simple.faces
            vertices = mesh_simple.vertices
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh_v_mean = mesh.vertices[mesh.faces].mean(axis=1)
        mesh_simple_v = mesh_simple.vertices
        tree = spatial.KDTree(mesh_v_mean)
        query = mesh_simple_v[faces].mean(axis=1)
        distance, index = tree.query(query)
        labels = labels[index].flatten()
        return mesh_simple, labels

    def _iterate_mesh_and_labels(self):
        root_mesh_folder = join(self.root, self.raw_folder)
        processed_dir = join(self.root, self.processed_folder)
        for root, dirs, files in os.walk(root_mesh_folder, followlinks=True):
            for file in files:
                if file.endswith(".obj"):
                    fn = file.replace('.obj', '')
                    if os.path.isfile(join(processed_dir, f'data_{fn}.pt')):
                        # already cached from a previous run - skip re-processing so the
                        # existing cached file is never touched
                        continue
                    json_path = join(root, file).replace('.obj', '.json')
                    if not os.path.isfile(json_path):
                        # raw_folder can contain meshes from other challenges/splits (e.g. a
                        # landmark-detection set) that never had segmentation labels - skip them
                        self._log(f'Skipping {fn}: no matching label file ({json_path})')
                        continue
                    mesh = trimesh.load(join(root, file))
                    with open(json_path) as f:
                        data = json.load(f)
                    labels = np.array(data["labels"])
                    labels = labels[mesh.faces]
                    labels = labels[:, 0]
                    labels = fdi_to_label(labels)
                    mesh, labels = self._donwscale_mesh(mesh, labels)
                    yield mesh, labels, fn

    def _check_cache_config(self):
        """
        Processing is additive (see _iterate_mesh_and_labels) and never deletes or overwrites existing
        cached files, so reusing a processed_folder with a different target_density/max_target_count
        than it was originally built with would silently mix two incompatible densities together. Record
        the config a folder was built with, and refuse to proceed if it doesn't match this instance's.
        """
        meta_path = join(self.root, self.processed_folder, '.cache_config.json')
        current = {'target_density': self.target_density, 'max_target_count': self.max_target_count}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                existing = json.load(f)
            if existing != current:
                raise ValueError(
                    f"processed_folder '{self.processed_folder}' was already built with "
                    f"{existing}, but this instance is configured with {current}. Processing never "
                    f"deletes or overwrites existing cached files, so reusing this folder would "
                    f"silently mix the two configs. Use a different processed_folder for {current}."
                )
        else:
            with open(meta_path, 'w') as f:
                json.dump(current, f)

    def _is_processed(self):
        files_processed = filter_files(join(self.root, self.processed_folder), 'pt')
        files_raw = filter_files(join(self.root, self.raw_folder), 'obj')
        return len(files_processed) == len(files_raw)

    def _process(self):
        self._log('Processing data')
        for mesh, labels, fn in self._loop(self._iterate_mesh_and_labels()):
            mesh = self.move_to_origin(mesh)
            data = process_mesh(mesh, torch.from_numpy(labels).long())
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            with open(f'{join(self.root, self.processed_folder)}/data_{fn}.pt', 'wb') as f:
                pickle.dump(data, f)
        self._log('Processing done')

    def _load_in_memory(self):
        files_processed = [join(self.root, self.processed_folder, f) for f in self.file_names]
        for i, f in enumerate(files_processed):
            file = open(join(self.root, self.processed_folder, f), 'rb')
            data = pickle.load(file)
            if self.post_transform is not None:
                data = self.post_transform(data)
            self.in_memory_data.append(data)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.in_memory:
            return self.in_memory_data[index]
        else:
            f = self.file_names[index]
            file = open(join(self.root, self.processed_folder, f), 'rb')
            data = pickle.load(file)
            if self.post_transform is not None:
                data = self.post_transform(data)
            return data











