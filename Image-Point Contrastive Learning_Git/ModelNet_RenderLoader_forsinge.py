import numpy as np
import torch.utils.data as data
from models.mvtn import *
from models.renderer import *
import os
from tqdm import tqdm
import warnings
from PIL import Image
import json
import torchvision.transforms as transforms
from pytorch3d.ops import sample_points_from_meshes

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

def ModelNet40Render(split, maskid = 19):
    y = []
    data_list = []
    nb_points = 1024
    data_dir = "data/ModelNet40/"

    if maskid == -1:
        output_dir = "/root/autodl-tmp/DataSet/ModelNet40_Multimodals/"
    else:
        output_dir = "/root/autodl-tmp/DataSet/ModelNet40_Multimodals_RandMask_single/"

    simplified_mesh = True
    cleaned_mesh = True
    dset_norm = 1
    initial_angle = -90

    classes, class_to_idx = find_classes(data_dir)

    is_rotated = False
    nb_views = 10
    image_size = 224
    object_color = "white"
    background_color =  "white"
    bug_names = ""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open (os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f)

    root = os.path.join(output_dir, split)
    if not os.path.exists(root):
        os.makedirs(os.path.join(root))

    process = True  # 没有文件，需要处理

    if process: #需处理文件
        print('Processing ' + str(split) +' data (only running in the first time)')
        if not os.path.exists(os.path.join(root, "pcs")):
            os.makedirs(os.path.join(root, "pcs"))

        # 创建图像文件夹
        if not os.path.exists(os.path.join(output_dir, split, "imgs")):
            os.makedirs(os.path.join(output_dir, split, "imgs"))

        for label in os.listdir(data_dir):
            for item in os.listdir(data_dir + '/' + label + '/' + split):
                if item.endswith(".off"):
                    y.append(class_to_idx[label])
                    data_list.append(data_dir + '/' + label + '/' + split + '/' + item)

        simplified_data_list = [file_name.replace(".off", "_SMPLER.obj") for file_name in data_list if file_name[-4::] == ".off"]
        points_list = [file_name.replace(".off", "POINTS.pkl") for file_name in data_list if file_name[-4::] == ".off"]
        data_list, simplified_data_list, y, points_list = sort_jointly([data_list, simplified_data_list, y, points_list], dim=0)

        if is_rotated:
            df = pd.read_csv(os.path.join(data_dir, "..", "rotated_modelnet_{}.csv".format(split)), sep=",")
            rotations_list = [df[df.mesh_path.isin([x])].to_dict("list") for x in data_list]

        correction_factors = [1]*len(data_list)

        if cleaned_mesh:
            fault_mesh_list = load_text(os.path.join(
                data_dir, "..", "{}_faults.txt".format(split)))
            fault_mesh_list = [int(x) for x in fault_mesh_list]
            for x in fault_mesh_list:
                correction_factors[x] = -1

        for index in tqdm(range(len(data_list)), total=len(data_list)):
            savefile_name = os.path.basename(data_list[index]).replace(".off", "")
            if savefile_name != 'cup_0094':
                continue
            if maskid == -1:
                save_path_pcs_fan = os.path.join(root, "pcs", savefile_name + '_fan.txt' )
                save_path_pcs = os.path.join(root, "pcs", savefile_name + '.txt')
            else:
                save_path_pcs_fan = os.path.join(root, "pcs", savefile_name + '_Mask' + str(maskid) + '_fan.txt')
                save_path_pcs = os.path.join(root, "pcs", savefile_name + '_Mask' + str(maskid) + '.txt')

            #if maskid != -1 and y[index] not in [22, 20, 25, 4]:
            #        continue

            if not simplified_mesh:
                threeobject = trimesh.load(data_list[index])
            else:
                threeobject = trimesh.load(simplified_data_list[index])

            if not is_rotated:
                angle = initial_angle
                rot_axis = [1, 0, 0]
            else:
                angle = rotations_list[index]["rot_theta"][0]
                rot_axis = [rotations_list[index]["rot_x"]
                            [0], rotations_list[index]["rot_y"][0], rotations_list[index]["rot_z"][0]]

            verts = np.array(threeobject.vertices.data.tolist())
            faces = np.array(threeobject.faces.data.tolist())
            if correction_factors[index] == -1 and cleaned_mesh and simplified_mesh:
                faces[:, 0], faces[:, 2] = faces[:, 2], faces[:, 0]

            verts = rotation_matrix(rot_axis, angle).dot(verts.T).T
            verts = torch_center_and_normalize(torch.from_numpy(verts).to(torch.float), p=dset_norm)
            faces = torch.from_numpy(faces)

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
                faces_fan = faces[~mask]
                faces = faces[mask]
                used_verts_fan = torch.unique(faces_fan)
                used_verts = torch.unique(faces)
                verts_fan = verts[used_verts_fan]
                verts = verts[used_verts]
                # verts = torch_center_and_normalize(verts, p=dset_norm)

                for i, vert in enumerate(used_verts_fan):
                    faces_fan[faces_fan == vert] = i
                # 更新面的索引
                for i, vert in enumerate(used_verts):
                    faces[faces == vert] = i
                # Transform verts and faces

            verts_rgb_fan = torch.ones_like(verts_fan)[None]
            textures_fan = Textures(verts_rgb=verts_rgb_fan)
            mesh_fan = Meshes(
                verts=[verts_fan],
                faces=[faces_fan],
                textures=textures_fan
            )

            verts_rgb = torch.ones_like(verts)[None]
            textures = Textures(verts_rgb=verts_rgb)
            mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=textures
            )
            num_1 = int((len(verts_fan) /(len(verts_fan) + len(verts))) * nb_points)
            num_2 = int((len(verts) / (len(verts_fan) + len(verts))) * nb_points)
            Point_Verts_fan = torch.from_numpy(trimesh.Trimesh(vertices=verts_fan, faces=faces_fan).sample(num_1, False))
            Point_Verts = torch.from_numpy(trimesh.Trimesh(vertices=verts, faces=faces).sample(num_2, False))
            mvrenderer = MVRenderer(nb_views=nb_views, image_size=image_size,
                                    pc_rendering=False, object_color=object_color,
                                    background_color=background_color,
                                    faces_per_pixel=1, points_radius=0.006,
                                    points_per_pixel=1,
                                    light_direction= "random" ,
                                    cull_backfaces=False)

            # views_azim, views_elev, views_dist = mvtn (points.unsqueeze(0), c_batch_size=1)

            views_dist = (1.2 * torch.ones((10), dtype=torch.float, requires_grad=False)).unsqueeze(0)
            views_elev = torch.tensor([0, 90, 180, 270, 225, 225, 315, 315, 0, 0], dtype=torch.float, requires_grad=False).unsqueeze(0)
            views_azim = torch.tensor([0, 0, 0, 0, -45, 45, -45, 45, -90, 90], dtype=torch.float, requires_grad=False).unsqueeze(0)

            rendered_images, cameras =mvrenderer.forward(meshes=mesh, points = None, azim=views_azim, elev=views_elev, dist=views_dist)
            rendered_images_np = rendered_images.cpu().numpy()
            rendered_images_np = (rendered_images_np * 255).astype(np.uint8)

            if len(np.unique(rendered_images_np)) < 3: #渲染出现全白
                bug_names = bug_names + savefile_name + "\n"
                continue

            np.savetxt(save_path_pcs_fan, Point_Verts_fan.numpy(), fmt='%f', comments='')
            np.savetxt(save_path_pcs, Point_Verts.numpy(), fmt='%f', comments='')
            # with open(save_path_pcs, 'wb') as f:
            #    pickle.dump([Point_Verts, y[index]], f)

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

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

if __name__ == "__main__":
    for split in ["test"]:
        for maskid in range(20):
            ModelNet40Render(split, maskid)
            # ModelNet40Render(split)
