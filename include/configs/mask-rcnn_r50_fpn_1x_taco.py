_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/taco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# https://mmdetection.readthedocs.io/en/v2.2.1/getting_started.html
# How to visualize results, how to use pretrained network, very important to use pretrained detector, 
# and then fine tune it on custom data