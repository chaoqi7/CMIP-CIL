import torch.utils.data as data
from models.renderer import *
import os
from tqdm import tqdm
import warnings
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple

# 忽略所有的警告
warnings.filterwarnings('ignore')

def ShapeNetRender(
        data_dir,
        split,
        nb_points,
        maskid=19,
        synsets=None,
        version: int = 2,
        load_textures = False,
        texture_resolution: int = 4,
        dset_norm: str = "inf",
        simplified_mesh=False,
        nb_views=10,
        image_size=224,
        object_color="white",
        background_color="white"
    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """

        if maskid == -1:
            output_dir = "/root/autodl-tmp/DataSet/ShapeNet55_Multimodals_Seed1993_Task1/"
        else:
            output_dir = "/root/autodl-tmp/DataSet/ShapeNet55_Multimodals_RandMask_Seed1993_Task1/"

        save_paths = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        root = os.path.join(output_dir, split)
        if not os.path.exists(root):
            os.makedirs(os.path.join(root))

        process = True # not os.listdir(root) 没有文件，需要处理

        if process:  # 无需处理文件
            print('Processing ' + str(split) + ' data (only running in the first time)')
            if not os.path.exists(os.path.join(root, "pcs")):
                os.makedirs(os.path.join(root, "pcs"))

            # 创建图像文件夹
            if not os.path.exists(os.path.join(output_dir, split, "imgs")):
                os.makedirs(os.path.join(output_dir, split, "imgs"))

            if version not in [1, 2]:
                raise ValueError("Version number must be either 1 or 2.")
            model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"
            if simplified_mesh:
                model_dir = "models/model_normalized_SMPLER.obj"
            splits = pd.read_csv(os.path.join(data_dir, "shapenet_split.csv"), sep=",", dtype=str)

            dict_file = "shapenet_synset_dict_v%d.json" % version
            with open(os.path.join(data_dir, dict_file), "r") as read_dict:
                synset_dict = json.load(read_dict)

            synset_inv = {label: offset for offset, label in synset_dict.items()}
            num_to_idx = {list(synset_dict.keys())[i]: i for i in range(len(synset_dict))}
            label_to_idx = {synset_dict[list(synset_dict.keys())[i]]: i for i in range(len(synset_dict))}

            if synsets is not None:

                synset_set = set()
                for synset in synsets:
                    if (synset in synset_dict.keys()) and (
                        os.path.isdir(os.path.join(data_dir, synset))
                    ):
                        synset_set.add(synset)
                    elif (synset in synset_inv.keys()) and (
                        (os.path.isdir(os.path.join(data_dir, synset_inv[synset])))
                    ):
                        synset_set.add(synset_inv[synset])
                    else:
                        msg = (
                            "Synset category %s either not part of ShapeNetCore dataset "
                            "or cannot be found in %s."
                        ) % (synset, data_dir)
                        warnings.warn(msg)

            else:
                synset_set = {
                    synset
                    for synset in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, synset))
                    and synset in synset_dict
                }

            synset_not_present = set(
                synset_dict.keys()).difference(synset_set)
            [synset_inv.pop(synset_dict[synset])
             for synset in synset_not_present]

            if len(synset_not_present) > 0:
                msg = (
                    "The following categories are included in ShapeNetCore ver.%d's "
                    "official mapping but not found in the dataset location %s: %s"
                    ""
                ) % (version, data_dir, ", ".join(synset_not_present))
                warnings.warn(msg)

            for synset in synset_set:
                if synset in [ '03759954',  '03337140',  '02808440',  '02924116',  '03636649',  '03085013']:
                    print("---Processing " + synset + "---")
                    save_item = 0
                    for model in os.listdir(os.path.join(data_dir, synset)):
                        if not os.path.exists(os.path.join(data_dir, synset, model, model_dir)):
                            msg = (
                                "Object file not found in the model directory %s "
                                "under synset directory %s."
                            ) % (model, synset)

                            continue

                        found = splits[splits.modelId.isin([model])]["split"]
                        if len(found) > 0:
                            if found.item() in split:

                                savefile_name = synset_dict[synset] + "_" + str(save_item)
                                save_item = save_item + 1
                                if maskid == -1:
                                    save_path_pcs = os.path.join(root, "pcs", savefile_name + '_1024dpts.dat')
                                else:
                                    save_path_pcs = os.path.join(root, "pcs", savefile_name + '_Mask' + str(maskid) + '_1024dpts.dat')

                                if os.path.exists(save_path_pcs):
                                    continue

                                model_path = os.path.join(data_dir, synset, model, model_dir)
                                verts, faces, textures = _load_mesh(model_path, load_textures, texture_resolution)
                                verts = torch_center_and_normalize(verts.to(torch.float), p=dset_norm)

                                if maskid != -1:
                                    # Transform verts and faces
                                    num_faces = faces.shape[0]
                                    # 创建一个连续的随机mask
                                    mask_size1 = (int)(num_faces // 2)  # 假设我们想要保留一半的面
                                    mask_size2 = (int)(num_faces // 4)
                                    mask_size = random.randint(mask_size2, mask_size1)

                                    mask = create_random_mask(num_faces, mask_size)
                                    # 将numpy数组转换为torch张量
                                    mask = torch.tensor(mask)
                                    # 使用mask来选择面
                                    faces = faces[mask]
                                    used_verts = torch.unique(faces)
                                    verts = verts[used_verts]
                                    verts = torch_center_and_normalize(verts, p=dset_norm)
                                    # 更新面的索引
                                    for i, vert in enumerate(used_verts):
                                        faces[faces == vert] = i

                                verts_rgb = torch.ones_like(verts)[None]
                                textures = Textures(verts_rgb=verts_rgb)
                                mesh = Meshes(
                                    verts=[verts],
                                    faces=[faces],
                                    textures=textures
                                )
                                points = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy()).sample(nb_points, False)
                                points = torch.from_numpy(points).to(torch.float)
                                points = torch_center_and_normalize(points, p=dset_norm)

                                mvrenderer = MVRenderer(nb_views=nb_views, image_size=image_size,
                                                        pc_rendering=False, object_color=object_color,
                                                        background_color=background_color,
                                                        faces_per_pixel=1, points_radius=0.006,
                                                        points_per_pixel=1,
                                                        light_direction="random",
                                                        cull_backfaces=False)

                                views_dist = (1.2 * torch.ones((10), dtype=torch.float, requires_grad=False)).unsqueeze(0)
                                views_elev = torch.tensor([0, 90, 180, 270, 225, 225, 315, 315, 0, 0], dtype=torch.float,
                                                          requires_grad=False).unsqueeze(0)
                                views_azim = torch.tensor([0, 0, 0, 0, -45, 45, -45, 45, -90, 90], dtype=torch.float,
                                                          requires_grad=False).unsqueeze(0)

                                rendered_images, cameras = mvrenderer.forward(meshes=mesh, points=points, azim=views_azim, elev=views_elev, dist=views_dist)
                                rendered_images_np = rendered_images.cpu().numpy()
                                rendered_images_np = (rendered_images_np * 255).astype(np.uint8)
                                # save_path_pcs = os.path.join(root, "pcs", savefile_name + '_%dpts.dat' % (nb_points))

                                with open(save_path_pcs, 'wb') as f:
                                    pickle.dump([points, num_to_idx[synset]], f)
                                save_paths.append(save_path_pcs)

                                for i in range(rendered_images_np.shape[0]):
                                    # 获取单个图像数据
                                    img_data = rendered_images_np[i]
                                    img = Image.fromarray(img_data)
                                    # save_path_pcs = os.path.join(root, "imgs", savefile_name + '_%dview.png' % (i))
                                    if maskid == -1:
                                        save_path_pcs = os.path.join(root, "imgs", savefile_name + '_%dview.png' % (i))
                                    else:
                                        save_path_pcs = os.path.join(root, "imgs", savefile_name + '_Mask' + str(maskid) + '_%dview.png' % (i))
                                    img.save(save_path_pcs)

def create_random_mask(length, mask_size):
    # 确保mask_size不大于length
    assert mask_size <= length
    # 随机选择mask的起始位置
    start = np.random.randint(length - mask_size + 1)
    # 创建一个全为False的数组
    mask = np.zeros(length, dtype=bool)
    # 将mask的位置设为True
    mask[start:start + mask_size] = True
    return mask

def _load_mesh(model_path, load_textures, texture_resolution) -> Tuple:
    from pytorch3d.io import load_obj

    verts, faces, aux = load_obj(
        model_path,
        create_texture_atlas=load_textures,
        load_textures=load_textures,
        texture_atlas_size=texture_resolution,
    )
    if load_textures:
        textures = aux.texture_atlas

    else:
        textures = verts.new_ones(
            faces.verts_idx.shape[0],
            texture_resolution,
            texture_resolution,
            3,)

    return verts, faces.verts_idx, textures

if __name__ == "__main__":

    data_dir = "data/ShapeNetCore.v2/"
    # split = "train"
    nb_points = 1024
    simplified_mesh = False
    cleaned_mesh = True
    process_data = True

    for split in ["train","val", "test"]:
        for maskid in range(20):
            print('--------Processing: '+ split + ' Mask: ' + str(maskid)+'--------')
            ShapeNetRender(data_dir, split, nb_points=nb_points, maskid=maskid, load_textures=True, dset_norm=1, simplified_mesh=simplified_mesh)