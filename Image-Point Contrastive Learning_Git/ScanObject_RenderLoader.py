import torch.utils.data as data
from models.mvtn import *
from models.renderer import *
import os
from tqdm import tqdm
import warnings
from PIL import Image
import json
import torchvision.transforms as transforms
import h5py
import pandas as pd
import open3d as o3d

# 忽略所有的警告
warnings.filterwarnings('ignore')

def reshape_point_cloud_to_cube(point_cloud, cube_size=14):
    # point_cloud shape: [N, 3]
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)

    # Create a grid of cubes
    x = np.linspace(min_coords[0], max_coords[0], cube_size)
    y = np.linspace(min_coords[1], max_coords[1], cube_size)
    z = np.linspace(min_coords[2], max_coords[2], cube_size)
    grid = np.stack(np.meshgrid(x, y, z), -1)  # [cube_size, cube_size, cube_size, 3]

    # Reshape the grid to [cube_size^3, 3]
    grid = grid.reshape(-1, 3)

    # For each point in the point cloud, find the closest cube
    dists = np.linalg.norm(point_cloud[:, None, :] - grid[None, :, :], axis=-1)  # [N, cube_size^3]
    closest_cube_indices = dists.argmin(axis=-1)

    # For each cube, average the coordinates of the points that are closest to it
    cubes = []
    for i in range(cube_size ** 3):
        points_in_cube = point_cloud[closest_cube_indices == i]
        if points_in_cube.shape[0] > 0:
            cube = points_in_cube.mean(axis=0)
        else:
            cube = np.zeros(3)
        cubes.append(cube)

    # Reshape the list of cubes to a [cube_size, cube_size, cube_size, 3] tensor
    cubes = np.stack(cubes).reshape(cube_size, cube_size, cube_size, 3)

    return cubes

def create_random_mask(length, mask_size):
    # 确保mask_size不大于length
    assert mask_size <= length
    # 随机选择mask的起始位置
    start = np.random.randint(length - mask_size + 1)
    # 创建一个全为False的数组
    mask = np.zeros(length, dtype=bool)
    # 将mask的位置设为True
    mask[start:start+mask_size] = True
    return mask

def ScanObjectRender(maskid = 19):
    nb_points = 10000
    data_dir = '/root/autodl-tmp/DataSet/ScanObjectNN_h5_files/h5_files/main_split/'
    # output_dir = "/root/autodl-tmp/DataSet/ScanObjectNN_Multimodals/"
    if maskid == -1:
        output_dir = "/root/autodl-tmp/DataSet/ScanObjectNN_Multimodals/"
    else:
        output_dir = "/root/autodl-tmp/DataSet/ScanObjectNN_Multimodals_RandMask/"
    dset_norm = 1
    classes = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']

    class_to_idx = {'bag': 0,  'bin' : 1, 'box' : 2, 'cabinet':3, 'chair':4, 'desk':5, 'display':6, 'door':7, 'shelf':8, 'table':9, 'bed': 10,
                    'pillow':11, 'sink':12, 'sofa':13,  'toilet':14}

    train_load_path = os.path.join(data_dir, 'training_objectdataset_augmentedrot_scale75.h5')
    test_load_path = os.path.join(data_dir, 'test_objectdataset_augmentedrot_scale75.h5')
    train_file = h5py.File(train_load_path, 'r')
    test_file = h5py.File(test_load_path, 'r')
    train_data = np.array(train_file['data']).astype(np.float32)
    train_targets = np.array(train_file['label']).astype(int)
    test_data = np.array(test_file['data']).astype(np.float32)
    test_targets = np.array(test_file['label']).astype(int)

    nb_views = 10
    image_size = 224
    object_color = "white"
    background_color =  "white"
    bug_names = ""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open (os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f)

    for split in ["train", "test"]:
        root = os.path.join(output_dir, split)
        if not os.path.exists(root):
            os.makedirs(os.path.join(root))
        if not os.path.exists(os.path.join(root, "pcs")):
            os.makedirs(os.path.join(root, "pcs"))
        if not os.path.exists(os.path.join(root, "imgs")):
            os.makedirs(os.path.join(root, "imgs"))

        print('Processing data (only running in the first time)')
        Data, Targets = None, None
        if split == "train":
            Data = train_data
            Targets = train_targets
        else:
            Data = test_data
            Targets = test_targets

        for index in tqdm(range(len(Targets)), total=len(Targets)):
            root = os.path.join(output_dir, split)
            savefile_name = classes[Targets[index]] + '_' + str(index)
            # save_path_pcs = os.path.join(root, "pcs", savefile_name + '_cubes.dat' )
            if maskid == -1:
                save_path_pcs = os.path.join(root, "pcs", savefile_name + '.dat' )
            else:
                save_path_pcs = os.path.join(root, "pcs", savefile_name + '_Mask' + str(maskid) + '.dat')

            if os.path.exists(save_path_pcs):
                continue

            points = Data[index]
            if len(points) == 0:
                continue

            points = torch_center_and_normalize(torch.from_numpy(points).to(torch.float), p=dset_norm)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
            pcd.estimate_normals()
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            verts = torch.tensor(mesh.vertices, dtype=torch.float32)  # (num_verts, 3)
            faces = torch.tensor(mesh.triangles, dtype=torch.int64)

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

                if len(verts) == 0:
                    continue
                verts = torch_center_and_normalize(verts, p=dset_norm)

                # 更新面的索引
                for i, vert in enumerate(used_verts):
                    faces[faces == vert] = i

            verts_rgb = torch.ones_like(verts)[None]
            textures = Textures(verts_rgb=verts_rgb)
            meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

            indices = np.random.choice(verts.shape[0], size=min(verts.shape[0], nb_points), replace=False)
            # Cubes = reshape_point_cloud_to_cube(points[indices])
            Cubes = points[indices]
            '''
            mvtn = MVTN(nb_views, views_config="spherical",
                        canonical_elevation=30.0,
                        canonical_distance=2.2,
                        shape_features_size=40, transform_distance=False,
                        input_view_noise=0.0, shape_extractor="PointNet",
                        screatch_feature_extractor=False).cuda()
            '''
            mvrenderer = MVRenderer(nb_views=nb_views, image_size=image_size,
                                    pc_rendering=False, object_color=object_color,
                                    background_color=background_color,
                                    faces_per_pixel=1, points_radius=0.006,
                                    points_per_pixel=1,
                                    light_direction= "random" ,
                                    cull_backfaces=False)

            #views_azim, views_elev, views_dist = mvtn (points.unsqueeze(0), c_batch_size=1)

            views_dist = (1.2 * torch.ones((10), dtype=torch.float, requires_grad=False)).unsqueeze(0)
            views_elev = torch.tensor([0, 90, 180, 270, 225, 225, 315, 315, 0, 0], dtype=torch.float, requires_grad=False).unsqueeze(0)
            views_azim = torch.tensor([0, 0, 0, 0, -45, 45, -45, 45, -90, 90], dtype=torch.float, requires_grad=False).unsqueeze(0)

            rendered_images, cameras =mvrenderer.forward(meshes=meshes, points = None, azim=views_azim.cuda(), elev=views_elev.cuda(), dist=views_dist.cuda())
            rendered_images_np = rendered_images.cpu().numpy()
            rendered_images_np = (rendered_images_np * 255).astype(np.uint8)

            if len(np.unique(rendered_images_np)) < 3: #渲染出现全白
                bug_names = bug_names + savefile_name + "\n"
                continue

            with open(save_path_pcs, 'wb') as f:
                pickle.dump([Cubes, Targets[index]], f)

            for i in range(rendered_images_np.shape[0]):
                # 获取单个图像数据
                img_data = rendered_images_np[i]
                img = Image.fromarray(img_data)
                if maskid == -1:
                    save_path_pcs = os.path.join(root, "imgs", savefile_name + '_%dview.png' % (i))
                else:
                    save_path_pcs = os.path.join(root, "imgs", savefile_name + '_Mask' + str(maskid) + '_%dview.png' % (i))
                img.save(save_path_pcs)
                ''''''
    print(bug_names)

if __name__ == "__main__":
    for maskid in range(20):
        ScanObjectRender(maskid)
