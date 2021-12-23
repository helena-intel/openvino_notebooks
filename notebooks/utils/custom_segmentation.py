# SegmentationModel implementation based on
# https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/models
import sys
sys.path.append("../utils")

import cv2
import numpy as np
from os import PathLike
from models import model
from notebook_utils import segmentation_map_to_overlay
import matplotlib
from pathlib import Path

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def resize_to_max(image_shape, max_shape):
    if max(image_shape) <= max_shape:
        new_shape = image_shape
    else:
        index = np.argmax(image_shape)
        factor = max_shape / image_shape[index]
        height, width = image_shape[:2]
        new_shape = int(factor * height), int(factor*width)
        return new_shape

def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def calc_avg_brightness(img):
    # Copyright (c) 2020 PaddlePaddle Authors. Licensed under the Apache License, Version 2.0
    R = img[..., 0].mean()
    G = img[..., 1].mean()
    B = img[..., 2].mean()

    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R


def adjust_brightness(dst, src):
    # Copyright (c) 2020 PaddlePaddle Authors. Licensed under the Apache License, Version 2.0
    brightness1, B1, G1, R1 = calc_avg_brightness(src)
    brightness2, B2, G2, R2 = calc_avg_brightness(dst)
    brightness_difference = brightness1 / brightness2
    dstf = dst * brightness_difference
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)
    return dstf


def resize_to_image_shape(result_image, original_image):
    height, width = original_image.shape[:2]
    result_image = cv2.resize(result_image, (width, height))
    return result_image


class MonodepthModel(model.Model):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        colormap: str = "viridis",
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=False,
    ):
        """
        Segmentation Model for use with Async Pipeline

        :param model_path: path to IR model .xml file
        :param colormap: array of shape (num_classes, 3) where colormap[i] contains the RGB color
            values for class i. Optional for binary segmentation, required for multiclass
        :param resize_shape: if specified, reshape the model to this shape
        :param sigmoid: if True, apply sigmoid to model result
        :param argmax: if True, apply argmax to model result
        :param rgb: set to True if the model expects RGB images as input
        """
        super().__init__(ie, model_path)
        self.colormap = colormap
        self.net = ie.read_network(model_path)
        self.output_layer = next(iter(self.net.outputs))
        self.input_layer = next(iter(self.net.input_info))
        if resize_shape is not None:
            self.net.reshape({self.input_layer: resize_shape})
        self.image_height, self.image_width = self.net.input_info[
            self.input_layer
        ].tensor_desc.dims[2:]

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.
        """
        meta = {}
        image = inputs[self.input_layer]
        meta["frame"] = image
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height))
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)
        return {self.input_layer: input_image}, meta

    def postprocess(self, outputs, preprocess_meta):
        """
        Convert raw network results into an RGB segmentation map with overlay
        """
        cmap = matplotlib.cm.get_cmap(self.colormap)
        result = outputs[self.output_layer]
        result = result.squeeze(0)
        result = normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        result = resize_to_image_shape(result, preprocess_meta["frame"])

        return result


class SegmentationModel(model.Model):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        colormap: np.ndarray = None,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=False,
        normalize=False
    ):
        """
        Segmentation Model for use with Async Pipeline

        :param model_path: path to IR model .xml file
        :param colormap: array of shape (num_classes, 3) where colormap[i] contains the RGB color
            values for class i. Optional for binary segmentation, required for multiclass
        :param resize_shape: if specified, reshape the model to this shape
        :param sigmoid: if True, apply sigmoid to model result
        :param argmax: if True, apply argmax to model result
        :param rgb: set to True if the model expects RGB images as input
        :param normalize: if True, divide input images by 255
        """
        super().__init__(ie, model_path)
        self.sigmoid = sigmoid
        self.argmax = argmax
        self.rgb = rgb
        self.normalize = normalize
        self.model_path = Path(model_path)
        self.net = ie.read_network(self.model_path)
        self.output_layer = next(iter(self.net.outputs))
        self.input_layer = next(iter(self.net.input_info))
        if resize_shape is not None:
            self.net.reshape({self.input_layer: resize_shape})
        self.image_height, self.image_width = self.net.input_info[
            self.input_layer
        ].tensor_desc.dims[2:]
        if colormap is None and self.net.outputs[self.output_layer].shape[1] == 1:
            self.colormap = np.array([[0, 0, 0], [0, 0, 255]])
        else:
            self.colormap = colormap
        # if self.colormap is None:
        #     raise ValueError("Please provide a colormap for multiclass segmentation")

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.
        """
        meta = {}
        image = inputs[self.input_layer]
        meta["frame"] = image.copy()
        if self.normalize:
            image = image.astype(np.float32) / 255
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(image, (self.image_width, self.image_height))
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)
        return {self.input_layer: input_image}, meta


    def postprocess(self, outputs, preprocess_meta):
        """
        Convert raw network results into an RGB segmentation map with overlay
        """
        alpha = 0.4

        if preprocess_meta["frame"].shape[-1] == 3:
            rgb_frame = preprocess_meta["frame"]
            if self.rgb:
                # reverse color channels to convert to BGR
                rgb_frame = rgb_frame[:, :, (2, 1, 0)]
        else:
            # Create RGB image by repeating channels in one-channel image
            rgb_frame = np.repeat(np.expand_dims(preprocess_meta["frame"], -1), 3, 2)
        res = outputs[self.output_layer].squeeze()

        result_mask_ir = sigmoid(res) if self.sigmoid else res

        if self.argmax:
            result_mask_ir = np.argmax(res, axis=0).astype(np.uint8)
        else:
            result_mask_ir = result_mask_ir.round().astype(np.uint8)
        overlay = segmentation_map_to_overlay(
            rgb_frame, result_mask_ir, alpha, colormap=self.colormap
        )

        return overlay


class U2NetModel(SegmentationModel):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=True,
    ):
        super().__init__(ie, model_path, rgb=True)#, colormap=None, resize_shape, sigmoid, argmax, rgb)

    def postprocess(self, outputs, preprocess_meta):
        result = outputs[self.output_layer].squeeze()
        result = np.rint(result)
        result_image = result.astype(np.uint8) * 255
        result_image = resize_to_image_shape(result_image, preprocess_meta["frame"])
        bg_removed_result = preprocess_meta["frame"].copy()
        bg_removed_result[result_image == 0] = 255
        return bg_removed_result


class PaddleAnimeModel(SegmentationModel):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=False,
    ):
        super().__init__(ie, model_path, rgb=True)#, colormap=None, resize_shape, sigmoid, argmax, rgb)


    def postprocess(self, outputs, preprocess_meta):
        result = outputs[self.output_layer].squeeze()
        result_image = (result * 0.5 + 0.5) * 255
        result_image = result_image.transpose((1, 2, 0))
        #height, width = preprocess_meta["frame"].shape[:2]
        #result_image = cv2.resize(result_image, (width, height))
        result_image = resize_to_image_shape(result_image, preprocess_meta["frame"])
        result_image = adjust_brightness(result_image, preprocess_meta["frame"])
        return result_image
    
    
class PaddleSuperResolutionModel(SegmentationModel):
    def __init__(
        self,
        ie,
        model_path: PathLike,
        resize_shape=None,
        sigmoid=False,
        argmax=False,
        rgb=True,
        normalize=True
    ):
        super().__init__(ie, model_path, resize_shape=resize_shape, rgb=True, normalize=True)#, colormap=None, resize_shape, sigmoid, argmax, rgb)

    def preprocess(self, inputs):
        """
        Resize the image to network input dimensions and transpose to
        network input shape with N,C,H,W layout.
        """
        meta = {}
        image = inputs[self.input_layer]
        meta["frame"] = image.copy()
        if self.normalize:
            image = image.astype(np.float32) / 255
        h, w = image.shape[:2]
        resize_shape = [1, 3, h, w]
        print(resize_shape)
        self.net.reshape({self.input_layer: resize_shape})
        if len(image.shape) == 3:
            input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        else:
            input_image = np.expand_dims(np.expand_dims(image, 0), 0)       
        return {self.input_layer: input_image}, meta

    def postprocess(self, outputs, preprocess_meta):
        result = outputs[self.output_layer].squeeze()
        print(result.shape)
        result_image = (result * 255).clip(0, 255).astype("uint8").transpose((1, 2, 0))
        return result_image
