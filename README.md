# CMIP-CIL:
A cross-modal benchmark for image-point class incremental learning.

## 📖Content
- **[Task of image-point class incremental learning (IP-CIL)](#IP-CIL)**
- **[Pretrained Models](#Pretrained-Models)**
- **[Code](#Code)**

## 🎨IP-CIL
- **Task 1: learns to classify objects in images, testing with point-cloud-based classifications. Task 2: learn new objects in images, testing point cloud ones in the current and former classes—the same in the following tasks.**
 
<img src="https://hv.z.wiki/autoupload/20250317/H6HN/930X876/IP-CIL.png" width="400" />

  
## 🌈Pretrained Models
**The pre-trained models are available** [[LINK](https://pan.baidu.com/s/1D1UzXUP5o-7L-tmTi6ONHA )] (CODE: 7g35).

## 🔍Experiments

- **Comparisons on ShapeNet55** (18 exemplar samples per class)

| Model | ${\mathcal{A}_b}$ | $\bar{\mathcal{A}} $ |
|--|--|--|
| LwF | 39.5 | 63.4 |
| iCaRL|44.6| 69.5 |
| RPS-Net | 63.5 | 78.4 |
| BiC | 64.2 | 78.8 |
| I3DOL | 67.3 | 81.6 |
| InOR-Net | 69.4 | 83.7 |
| Ours | **83.4** | **89.3** |

![screenshot](https://cdn.z.wiki/autoupload/20241126/alYG/587X392/Experiment1.png)

- **Comparisons on ShapeNet55** (exemplar-free)

| Model | ${\mathcal{A}_b}$ | $\bar{\mathcal{A}} $ |
|--|--|--|
| FETRIL | 55.0 | 65.4 |
| 3D-EASE1 |52.2| 68.1 |
| 3D-EASE2 | 68.4 | 82.4 |
| Ours | **70.1** | **84.1** |

![screenshot](https://cdn.z.wiki/autoupload/20241126/qhVF/615X416/Experiment2.png)
  
## 💻Code

- **Run Experiment**
  
  The continual learning starts with [main_3DShape.py](./main_3DShape.py). <br>
  The experiment configurations can be set in [cil_config.json](./exps/cil_config.json).


- **Environment**
  
  python 3.8 <br>
  open3d 0.18 <br>
  scikit-learn 1.5.0 <br>
  torch 1.13.1+cu117 <br>
  GCC >= 4.9
  
- **Acknowledgments**

  The following repos provide helpful components/functions in our work: <br>
  [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) <br>
  [PyCIL](https://github.com/G-U-N/PyCIL) <br>
  [POINT-BERT](https://github.com/Julie-tang00/Point-BERT) <br>
  [CosineLinear](https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py)
