# CPGA

The dataset and code of the paper "CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement". 

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## 1.1 Environment
```python
conda create -n cpga python=3.7 -y && conda activate cpga

git clone --depth=1 https://github.com/VQE-CPGA/CPGA && cd VQE-CPGA/CPGA/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```
## 1.2 DCNv2
```python
cd ops/dcn/
bash build.sh
```
Check if DCNv2 work (optional)
```python
python simple_check.py
```
## 1.3 VCP dataset
**Download raw and compressed videos** 

Please check [BaiduPan](https://pan.baidu.com/s/1IFjZF2MvCyVOmgTBHgl2IA),Code [qix5].

**Edit YML**

You need to edit option_CPGA_vcp_#_QP#.yml file.

**Generate LMDB**

The LMDB generation for speeding up IO during training.
```python
python create_vcp.py --opt_path option_CPGA_vcp_#_QP#.yml
```
Finally, the VCP dataset root will be sym-linked to the folder ./data/ automatically.

## 1.4 Test dataset

We use the JCT-VC testing dataset in [JCT-VC](https://ieeexplore.ieee.org/document/6317156). Download raw and compressed videos [BaiduPan](https://pan.baidu.com/s/1IFjZF2MvCyVOmgTBHgl2IA),Code [qix5].

# Train
```python
python train_CPGA.py --opt_path ./config/option_CPGA_vcp_LDB_22.yml
```
# Test
```python
python test_CPGA.py --opt_path ./config/option_CPGA_vcp_LDB_22.yml
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@inproceedings{zhu2024cpga,
  title={CPGA: Coding Priors-Guided Aggregation Network for Compressed Video Quality Enhancement},
  author={Zhu, Qiang and Hao, Jinhua and Ding, Yukang and Liu, Yu and Mo, Qiao and Sun, Ming and Zhou, Chao and Zhu, Shuyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
}
@article{zhu2024deep,
  title={Deep Compressed Video Super-Resolution With Guidance of Coding Priors},
  author={Qiang Zhu, Feiyu Chen, Yu Liu, Shuyuan Zhu, Bing Zeng},
  journal={ IEEE Transactions on Broadcasting }
}
@article{zhu2024compressed,
  title={Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion},
  author={Qiang, Zhu and Yajun, Qiu and Yu, Liu and Shuyuan, Zhu and Bing, Zeng},
  journal={IEEE Signal Processing Letters}
}
@inproceedings{mo2025oapt,
  title={OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal},
  author={Mo, Qiao and Ding, Yukang and Hao, Jinhua and Zhu, Qiang and Sun, Ming and Zhou, Chao and Chen, Feiyu and Zhu, Shuyuan},
  booktitle={European Conference on Computer Vision}
}
```
# Related Works
We also released some compressed video quality enhancement models, e.g., [STDF](https://github.com/RyanXingQL/STDF-PyTorch), [RFDA](https://github.com/zhaominyiz/RFDA-PyTorch), [CF-STIF](https://github.com/xiaomingxige/CF-STIF), and  [STDR](https://github.com/xiaomingxige/STDR).

Our project is built on the [STDF](https://github.com/RyanXingQL/STDF-PyTorch). If there are some problems with the implementation, please refer to [STDF](https://github.com/RyanXingQL/STDF-PyTorch).
We adopt Apache License v2.0. For other licenses, please refer to [DCNv2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/master/LICENSE).
