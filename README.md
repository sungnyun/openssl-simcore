# OpenSSL-SimCore (CVPR 2023)
[arXiv](https://arxiv.org/abs/2303.11101) | [Video](https://www.youtube.com/watch?v=f_-dIVRo8Q8) | [BibTeX](#bibtex)


<p align="center">
<img width="1394" src="https://user-images.githubusercontent.com/46050900/226794108-4ca0e8e8-0d1b-4509-97b5-214b41f03d7a.png">
</p>

[**Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning**](https://arxiv.org/abs/2303.11101)<br/>
[Sungnyun Kim](https://github.com/sungnyun)\*,
[Sangmin Bae](https://www.raymin0223.com)\*,
[Se-Young Yun](https://fbsqkd.github.io)<br/>
\* equal contribution

- **Open-set Self-Supervised Learning (OpenSSL) task**: an unlabeled open-set available during the pretraining phase on the fine-grained dataset.
- **SimCore**: simple coreset selection algorithm to leverage a subset semantically similar to the target dataset.
- SimCore significantly improves representation learning performance in various downstream tasks.


## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```


## Data Preparation
We used 11 fine-grained datasets and 7 open-sets.
Place each data files into `data/[DATASET_NAME]/` (it should be constructed as the `torchvision.datasets.ImageFolder` format).    
To download and setup the data, please see the [docs](data/README.md) and run python files, if necessary.
```bash
$ cd data/
$ python [DATASET_NAME]_image_folder_generator.py
```

## Pretraining
To simply pretrain the model, run the shell file. (We support multi-GPUs training, while we utilized 4 GPUs.)    
You will need to define the **path for each dataset**, and the **retrieval model checkpoint**. 
```bash
# specify $TAG and $DATA

$ CUDA_VISIBLE_DEVICES=<GPU_ID> bash run_selfsup.sh
```
Here are some important arguments to be considered.
- `--dataset1`: fine-grained target dataset name
- `--dataset2`: open-set name (default: imagenet)
- `--data_folder1`: directory where the `dataset1` is located
- `--data_folder2`: directory where the `dataset2` is located
- `--retrieval_ckpt`: retrieval model checkpoint before SimCore pretraining; for this, pretrain vanilla SSL for 1K epochs
- `--model`: model architecture (default: resnet50), see [models](models/)
- `--method`: self-supervised learning method (default: simclr), see [ssl](ssl/)
- `--sampling_method`: strategy for sampling from the open-set (choose between "random" or "simcore")
- `--no_sampling`: if sampling unwanted (vanilla SSL pretrain), set this True

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/`. For example, if you run the default shell file, the last epoch checkpoint will be saved as `save/$DATA_resnet50_pretrain_simclr_merge_imagenet_$TAG/last.pth`.


## Linear Evaluation
Linear evaluation of the pretrained models can be similarly implemented as the pretraining.    
Run the following shell file, with the **pretrained model checkpoint** additionally defined.
```bash
# specify $TAG, $DATA, and --pretrained_ckpt

$ CUDA_VISIBLE_DEVICES=<GPU_ID> bash run_sup.sh
```
We also support **kNN evaluation** (`--knn`, `--topk`) and **semi-supervised fine-tuning** (`--label_ratio`, `--e2e`).

### Result
SimCore with a stopping criterion highly improves the accuracy by +10.5% (averaged over 11 datasets), compared to the pretraining without any open-set.
<p align="center">
<img width="750" src="https://user-images.githubusercontent.com/46050900/226905308-9cec7d37-f06e-4b6d-8a49-370ea6394afa.png">
</p>

### Try other open-sets
SimCore works with various, or even uncurated open-sets. You can also try with your custom, web-crawled, or uncurated open-sets.
<p align="center">
<img width="350" src="https://user-images.githubusercontent.com/46050900/226906525-6fcda233-692d-48e9-a241-2faa3daf3893.png">
&nbsp;
&nbsp;
&nbsp;
<img width="350" src="https://user-images.githubusercontent.com/46050900/226906595-2f3e293e-1f79-4992-bc22-fd128cb131d9.png">
</p>


## Downstream Tasks
SimCore is extensively evaluated in various downstream tasks.    
We thus provide the training and evaluation codes for following downstream tasks.    
For more details, please see the [docs](downstream/README.md) and `downstream/` directory.    
- [object detection](downstream/detection)
- [pixel-wise segmentation](downstream/segmentation)
- [open-set semi-supervised learning](downstream/opensemi)
- [webly supervised learning](downstream/weblysup)
- [semi-supervised learning](downstream/semisup)
- [active learning](downstream/active)
- [hard negative mining](downstream/hnm)

 Use the pretrained model checkpoint to run each downstream task.


## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{kim2023coreset,
  title={Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning},
  author={Kim, Sungnyun and Bae, Sangmin and Yun, Se-Young},
  journal={arXiv preprint arXiv:2303.11101},
  year={2023}
}
```

## Contact
- Sungnyun Kim: ksn4397@kaist.ac.kr
- Sangmin Bae: bsmn0223@kaist.ac.kr
