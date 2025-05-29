# 项目说明
## 1. 环境
参照mmdetection官方文档安装mmdetection及环境
## 2. 数据集
### 下载
下载 VOC2007 trainval 数据

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

下载 VOC2007 test 数据

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

解压

tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar

### 转换为 COCO 格式
cd mmdetection

python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o data/VOCdevkit --out-format coco

## 3. 训练
### 训练 mask-rcnn
python tools/train.py configs/voc/mask-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/mask-rcnn_r50_fpn_1x_voc --cfg-options randomness.seed=91

### 训练 sparse-rcnn
python tools/train.py configs/voc/sparse-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/sparse-rcnn_r50_fpn_1x_voc --cfg-options randomness.seed=91

## 4. 推理
### 使用现有模型推理
1. 模型下载

Sparse-RCNN(mAP_50:0.612):https://drive.google.com/file/d/1ce5pslacvBQ_oVbTIh6fg2D1amTq8bEW/view?usp=drive_link

Mask-RCNN(mAP_50:0.706):https://drive.google.com/file/d/1HUwHLG5mG13xu28wAvqiH3jLdLEq7h30/view?usp=drive_link

2. 推理

Sparse-RCNN:python tools/test.py configs/voc/sparse-rcnn_r50_fpn_1x_coco.py work_dirs/sparse-rcnn_r50_fpn_1x_voc/latest.pth --show-dir work_dirs/sparse-rcnn_r50_fpn_1x_voc/vis_data

Mask-RCNN:python tools/test.py configs/voc/mask-rcnn_r50_fpn_1x_coco.py work_dirs/mask-rcnn_r50_fpn_1x_voc/latest.pth --show-dir work_dirs/mask-rcnn_r50_fpn_1x_voc/vis_data
