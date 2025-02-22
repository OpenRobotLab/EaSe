<div align="center">
<h1>Evolving Symbolic 3D Visual Grounder with Weakly Supervised Reflection</h1>
  
Boyu Mi,  Hanqing Wang, Tai Wang, Yilun Chen, Jiangmiao Pang

Shanghai AI Laboratory

<a href="https://arxiv.org/abs/2502.01401"><img src='https://img.shields.io/badge/arXiv-Paper-red20B2AA?style=for-the-badge' alt='Paper'></a>
<a href='https://huggingface.co/datasets/miboyu5/EaSe'><img src='https://img.shields.io/badge/huggingface-dataset-yellow?style=for-the-badge'  alt='Dataset'></a>
</div>

## Environment Installation

```
pip install -r requirements.txt
```

Set your openai api key:

```bash
export OPENAI_API_KEY=your_api_key
```

## Data Preparation

The `data/` dir should be organized as follows:

```
data
├── frames
│   ├── color
│   │   ├── 0.png
│   │   ├── 20.png
│   │   └── ...
├── referit3d
│   ├── annotations
│   ├── scan_data
├── symoblic_exp
│   ├── nr3d.jsonl
│   ├── scanrefer.json
├── test_data
│   ├── above
│   ├── behind
│   ├── ...
├── seg
├── nr3d_masks
├── scanrefer_masks
├── feats_3d.pkl
├── tables.pkl

```

- `frames`: RGB images of the scenes. [download_link](https://drive.google.com/file/d/1VVnj3DAcOWqZhB6Vi0gWdzA9gTKQwrej/view?usp=drive_link)
- `referit3d`: processed referit3d dataset from [vil3dref](https://www.dropbox.com/s/n0m5bpfvea1fg7w/referit3d.tar.gz?dl=0)
- `symbolic_exp`: symbolic expressiones.
- `test_data`: test data for code generation.
- `seg`: segmentation results of 3D point clouds for ScanRefer. [download_link](https://drive.google.com/file/d/1VRW_ew9Hwmsg-DRf22l_MHgFVB0UU1K0/view?usp=drive_link)
- `nr3d_masks`: 2D GT object masks. [download_link](https://drive.google.com/file/d/1Z0pRv_UV7P_aNHsYHVkUz-lLaMCU2C9i/view?usp=sharing)
- `scanrefer_masks`: 2D predicted object masks. [download_link](https://drive.google.com/file/d/1v4nqJSOFVh7MAmyDo92Xze01U00yr1bB/view?usp=drive_link)
- `feats_3d.pkl`: predicted object labels for Nr3D from [ZSVG3D](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019046_link_cuhk_edu_cn/ERMP88uTVCNLhzofKub7MsMBvaRAFXVr5abbQUjRYyYDiA?e=x6aKC9)
- `tables.pkl`: tables for code generation. [download_link](https://drive.google.com/file/d/11sN1ndS-DptYVH_xUXxaiWhwbvIaozuW/view?usp=drive_link)

[huggingface dataset](https://huggingface.co/datasets/miboyu5/EaSe)
## (Optional) Relation Encoder Generation

Run `src/relation_encoders/run_optim.py` to generate relation encoders for 6 relations:
`left`, `right`, `between`, `corner`, `above`, `below`, `behind`.

After the optimization is done, you will get the relation encoders and their accuracy on test cases under `data/test_data/{relation_name}/trajs`.
Then you can select the best relation encoders for evaluation.
You can also use the provided relation encoders in `src/relation_encoders`.

## (Optional) Features Computation

```bash
python -m src.relation_encoders.compute_features \
    --dataset scanrefer \
    --output $OUTPUT_DIR \
    --label pred
```

`--dataset` option can be `scanrefer` or `nr3d`. The `--label` option can be `gt` or `pred`.
Now we only support the `pred` label for ScanRefer because there is no GT label in standard evaluation protocols.

After running, you will get features in `.pth` format in the `$OUTPUT_DIR` directory.

You can also download our prepared features:
[nr3d(pred label)](https://drive.google.com/file/d/1iR-3bbewssQEDKWJg88ZRrpk3mRkYUz1/view?usp=drive_link)
[nr3d(gt label)](https://drive.google.com/file/d/1SpMZKAnhnDeQvvxfVxDQT-LYQPQc-TQJ/view?usp=drive_link)
[scanrefer](https://drive.google.com/file/d/1FR6ZKQURqyHLTlo-9dP4sKbayG4FDlxE/view?usp=drive_link)


## Evaluation

Nr3d Evaluation:

```bash
python -m src.eval.eval_nr3d \
    --features_path output/nr3d_features_per_scene_pred_label.pth \
    --top_k 5 \
    --threshold 0.9 \
    --label_type pred \
    --use_vlm 
```

ScanRefer Evaluation:

```bash
python -m src.eval.eval_scanrefer \
    --features_path output/scanrefer_features_per_scene.pth \
    --top_k 5 \
    --threshold 0.1 \
    --use_vlm
```

Change `features_path` and `label_type` if you'd like to evaluate on the ground truth labels.
Set `--use_vlm`, `--top_k` and threshold to use the VLM model for evaluation.
Please refer to our paper for the meanings of these parameters.

## Acknowledgement

Thank following repositories for their contributions:

- [ZSVG3D](https://github.com/CurryYuan/ZSVG3D)
- [ReferIt3D](https://github.com/referit3d/referit3d)
- [Vil3dref](https://github.com/cshizhe/vil3dref)

## Awesome Concurrent Works

- [SeeGround](https://seeground.github.io/)
- [CSVG](https://github.com/sunsleaf/CSVG)
