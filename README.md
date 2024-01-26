# MMROS Wrapper

## Overview
MMROS Wrapper is a ROS package that provides a simple node, `MMRosWrapper`, which subscribes to an image topic, performs instance segmentation using the `mmdetection` framework with the help of `mmdeploy`, and collects litter objects from the images. It uses `torch` for neural network computations.

## Dependencies
This package relies on the following external libraries and frameworks:

- [torch](https://pytorch.org/): PyTorch is an open-source machine learning library used for tensor computations and deep learning.
- [mmdeploy](https://github.com/open-mmlab/mmdeploy): MMDEPLOY is an efficient deep learning deployment toolbox built on top of PyTorch.
- [mmdetection](https://github.com/open-mmlab/mmdetection): MMDetection is an open-source object detection toolbox based on PyTorch.

Make sure to install these dependencies before running the ROS package.

## Launching the Node
To launch the `MMRosWrapper` node, use the provided launch file:

```
roslaunch mmros_wrapper launch_mmr_ros_wrapper.launch deploy_cfg_path:=path/to/deploy_cfg.yaml model_cfg_path:=path/to/model_cfg.yaml backend_model_name:=your_backend_model_name
```

Replace path/to/deploy_cfg.yaml and path/to/model_cfg.yaml with the appropriate file paths for your configuration. Also, provide the your_backend_model_name that corresponds to the backend model you want to use for instance segmentation.

## Instance Segmentation for Litter Collection with a Drone

Instance segmentation is a computer vision task that involves detecting and segmenting each individual object instance in an image. In this ROS package, the instance segmentation technique is utilized to detect and collect litter objects from images captured by a drone.

When the MMRosWrapper node is running, it subscribes to the specified image topic, processes the received images using the mmdetection framework, and identifies individual litter objects in the scene. This information can be further used to plan and execute litter collection actions with the drone.

Note: The ROS package and the node are provided as a basic example, and for real-world applications, additional components such as object detection and control logic for the drone would be needed to create a fully functional litter collection system.

Feel free to modify and extend this ROS package to suit your specific litter collection use case with a drone.

## SheBang! 

First line of the script `#!/usr/bin/python` heavily depends on which platform you use. 
Check out where's your python interpreter located before trying to run anything. 

