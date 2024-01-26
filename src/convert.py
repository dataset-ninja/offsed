import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
)
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = "/home/alex/DATASETS/TODO/OPEDD/left"
    anns_path = "/home/alex/DATASETS/TODO/OPEDD/devkit_offsed/SegmentationClass"
    batch_size = 30
    images_ext = ".png"
    ds_name = "ds"

    def get_unique_colors(img):
        unique_colors = []
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
        unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
        indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
        for col, indx in col2indx.items():
            if col != 0:
                unique_colors.append((col // (256**2), (col // 256) % 256, col % 256))

        return unique_colors

    def create_ann(image_path):
        labels = []

        img_height = 1242
        img_wight = 2208

        image_name = get_file_name_with_ext(image_path)

        mask_path = os.path.join(anns_path, image_name)
        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            mask_np = sly.imaging.image.read(mask_path)
            unique_colors = get_unique_colors(mask_np)
            for color in unique_colors:
                mask = np.all(mask_np == color, axis=2)
                ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    bitmap = sly.Bitmap(data=obj_mask)
                    if bitmap.area > 70:
                        obj_class = color_to_class[color]
                        label = sly.Label(bitmap, obj_class)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    building = sly.ObjClass("building", sly.Bitmap, color=(70, 70, 70))
    bush = sly.ObjClass("bush", sly.Bitmap, color=(99, 128, 52))
    camper = sly.ObjClass("camper", sly.Bitmap, color=(0, 0, 90))
    car = sly.ObjClass("car", sly.Bitmap, color=(0, 0, 142))
    crops = sly.ObjClass("crops", sly.Bitmap, color=(184, 151, 51))
    drivable_dirt = sly.ObjClass("drivable dirt", sly.Bitmap, color=(81, 0, 81))
    drivable_pavement = sly.ObjClass("drivable pavement", sly.Bitmap, color=(128, 64, 128))
    excavator = sly.ObjClass("excavator", sly.Bitmap, color=(0, 60, 100))
    grass = sly.ObjClass("grass", sly.Bitmap, color=(24, 226, 31))
    guard_rail = sly.ObjClass("guard rail", sly.Bitmap, color=(180, 165, 180))
    held_object = sly.ObjClass("held object", sly.Bitmap, color=(44, 231, 215))
    nondrivable_dirt = sly.ObjClass("nondrivable dirt", sly.Bitmap, color=(175, 44, 190))
    nondrivable_pavement = sly.ObjClass("nondrivable pavement", sly.Bitmap, color=(244, 35, 232))
    obstacle = sly.ObjClass("obstacle", sly.Bitmap, color=(255, 0, 0))
    person = sly.ObjClass("person", sly.Bitmap, color=(209, 43, 77))
    sky = sly.ObjClass("sky", sly.Bitmap, color=(70, 130, 180))
    tree = sly.ObjClass("tree", sly.Bitmap, color=(107, 142, 35))
    truck = sly.ObjClass("truck", sly.Bitmap, color=(0, 0, 70))
    wall = sly.ObjClass("wall", sly.Bitmap, color=(102, 102, 156))
    background = sly.ObjClass("background", sly.Bitmap, color=(0, 0, 0))

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[
            background,
            building,
            bush,
            camper,
            car,
            crops,
            drivable_dirt,
            drivable_pavement,
            excavator,
            grass,
            guard_rail,
            held_object,
            nondrivable_dirt,
            nondrivable_pavement,
            obstacle,
            person,
            sky,
            tree,
            truck,
            wall,
        ]
    )

    color_to_class = {}
    for curr_class in meta.obj_classes:
        color_to_class[curr_class.color] = curr_class
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_names = [
        im_name for im_name in os.listdir(images_path) if get_file_ext(im_name) == images_ext
    ]

    progress = sly.Progress("Add data to {} dataset".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [
            os.path.join(images_path, image_name) for image_name in img_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    return project
