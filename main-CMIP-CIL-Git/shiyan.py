import os
import pickle
import numpy as np

def convert_dat_to_xyz(dat_file_path, xyz_file_path):
    """将 .dat 文件转换为 .xyz 文件"""
    with open(dat_file_path, 'rb') as dat_file:
        pcs_data = pickle.load(dat_file)[0].numpy()
        np.savetxt(xyz_file_path, pcs_data, fmt='%f', header='X Y Z', comments='')

def process_directory(directory_path):
    """处理指定文件夹中的所有 .dat 文件"""
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.dat'):
            # 构建 .dat 文件的完整路径
            dat_file_path = os.path.join(directory_path, filename)

            # 构建输出的 .xyz 文件路径
            xyz_file_path = os.path.splitext(dat_file_path)[0] + '.xyz'

            # 转换文件
            convert_dat_to_xyz(dat_file_path, xyz_file_path)
            print(f"Converted {dat_file_path} to {xyz_file_path}")

if __name__ == '__main__':
    directory_path = '/root/autodl-tmp/mesh-to-image/'  # 替换为您的文件夹路径
    process_directory(directory_path)
