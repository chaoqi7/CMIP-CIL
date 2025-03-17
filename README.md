# CMIP-CIL:
A cross-modal benchmark for image-point class incremental learning.

## 📖Content
- **[Task of image-point class incremental learning (IP-CIL)](#IP-CIL)**
- **[Pretrained Models](#Pretrained-Models)**
- **[Code](#Code)**

## 🎨IP-CIL
- **Task 1: learns to classify objects in images, testing with point-cloud-based classifications. Task 2: learn new objects in images, testing point cloud ones in the current and former classes—the same in the following tasks.**
 
<img src="https://hv.z.wiki/autoupload/20250317/H6HN/930X876/IP-CIL.png" width="450" />

  
## 🌈Image-Point Contrastive Learning Models
**The pre-trained models are available** [[LINK](https://pan.baidu.com/s/1D1UzXUP5o-7L-tmTi6ONHA )] (CODE: 7g35).

## 🔍Experiments

- Comparisons on m-MN40-*Inc*.4

| Model | ${\mathcal{A}_B}$ | $\bar{\mathcal{A}} $ |
|--|--|--|
| iCaRL | 25.4| 48.4 |
| WA | 20.4 | 40.9 |
| PODNet | 29.0 | 51.9 |
| SimpleCIL | 36.1 | 50.2 |
| Ours | **50.8** | **63.4** |

<img src="https://hv.z.wiki/autoupload/20250317/2B4F/1005X630/result1-ab-modelnet.PNG" width="450" />

- Comparisons on m-SN55-*Inc*.6

| Model | ${\mathcal{A}_B}$ | $\bar{\mathcal{A}} $ |
|--|--|--|
| iCaRL | 38.5| 57.9 |
| WA | 24.5 | 43.1 |
| PODNet | 31.1 | 55.2 |
| SimpleCIL | 27.3 |  48.9 |
| Ours | **41.9** | **61.8** |

<img src="https://cdn.z.wiki/autoupload/20250317/7jHu/806X528/result2-ab-shapenet.PNG" width="450" />
  
## 💼DataSet

- **The multimodal dataset come from image rendering on the original point clouds.**
  
  The ShapeNet image rendering starts with [ShapeNet_RenderLoader.py](./Image-Point%20Contrastive%20Learning_Git/ShapeNet_RenderLoader.py). <br>
  The ModelNet image rendering starts with [ModelNet_RenderLoader.json](./Image-Point%20Contrastive%20Learning_Git/ModelNet_RenderLoader.py).

## 💻Code

- **Run Experiment**
  
  ShapeNet continual learning starts with [main_shapenet.py](./main-CMIP-CIL-Git/main_shapenet.py). <br>
  ModelNet continual learning starts with [main_modelnet.py](./main-CMIP-CIL-Git/main_modelnet.py).
  
- **Environment**
  
  python 3.8 <br>
  open3d 0.18 <br>
  scikit-learn 1.5.0 <br>
  torch 1.13.1+cu117 <br>
  GCC >= 4.9
  
- **Acknowledgments**

  The following repos provide helpful components/functions in our work: <br>
  [PyCIL](https://github.com/G-U-N/PyCIL) <br>
  [MVTN](https://github.com/ajhamdi/MVTN)
