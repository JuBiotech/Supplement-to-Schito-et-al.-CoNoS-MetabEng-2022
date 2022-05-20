import cv2
import torch
from cellpose import models
import numpy as np
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt


def pairwise_distances(points):
    distances = []

    if len(points) == 0:
        return distances

    for a, b in zip(points, points[1:]):
        distances.append(np.linalg.norm(a - b))

    return distances


def extract_rois(int_mask):
    num_cells = np.max(int_mask)
    score_threshold = 0.5

    all_contours = []

    for index in range(1, num_cells + 1):
        bool_mask = int_mask == index

        contours, hierarchy = cv2.findContours(
            np.where(bool_mask > score_threshold, 1, 0).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            contour = np.squeeze(contour)
            if contour.shape[0] < 3:
                # drop non 2D contours
                continue

            all_contours.append(contour)

    return all_contours


def segmentation(images, omni):

    # use_GPU = models.use_gpu()
    use_GPU = torch.cuda.is_available()
    print(">>> GPU activated? %d" % use_GPU)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    if omni:
        print("Loading omni model...")
        model = models.Cellpose(gpu=use_GPU, model_type="bact_omni", omni=True)
    else:
        model = models.Cellpose(gpu=use_GPU, model_type="cyto")

    channels = [[0, 0]]
    diameter = 30
    flow_threshold = 0.4
    cellprob_threshold = 0.2

    # flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold

    try:
        # masks, flows, styles, diams = model.eval(images, channels=channels, rescale=None, diameter=None, flow_threshold=.9, mask_threshold=.25, resample=True, diam_threshold=100)

        masks, flows, styles, diams = model.eval(
            images,
            channels=[0, 0],
            diameter=0.0,
            invert=False,
            net_avg=False,
            augment=False,
            resample=False,
            do_3D=False,
            progress=None,
            omni=True,
        )
    except:
        print("Error in OmniPose prediction")
        masks = [
            [],
        ]

    import cv2

    full_result = []

    for res in process_map(extract_rois, masks, max_workers=6, chunksize=4):
        full_result.append(res)

    """
    for i,_ in tqdm.tqdm(enumerate(images)):
        int_mask = masks[i]#

        num_cells = np.max(int_mask)
        score_threshold = 0.5

        all_contours = []

        for index in range(1, num_cells+1):
            bool_mask = int_mask == index

            contours, hierarchy = cv2.findContours(np.where(bool_mask > score_threshold, 1, 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = np.squeeze(contour)
                all_contours.append(contour)

        full_result.append(all_contours)
    """
    return full_result


import cv2
import numpy as np
import os.path as osp
import tqdm
import numpy.ma as ma
from PIL import Image, ImageDraw
import pandas as pd
import itertools
from tqdm.contrib.concurrent import process_map

from functools import partial

import getpass


def unpack(data, func):
    return func(*data)


def extract_fluorescence(images, rois):
    datapoints = []

    print("Extract fluorescence...")

    # apply fluorescence analysis in parallel to all images
    for frame, data in enumerate(
        process_map(
            partial(unpack, func=analyze_fluorescence),
            zip(images, rois),
            max_workers=6,
            chunksize=4,
        )
    ):
        # extract and store result data
        for r, g in data:
            datapoints.append((frame, r, g))

    # build pandas dataframe from datapoints
    df = pd.DataFrame(datapoints, columns=["frame", "red", "green"])

    # store to files
    # df.to_csv(osp.join(basepath, 'datapoints.csv'))
    # df.to_pickle(osp.join(basepath, 'datapoints.pkl'))

    return df


"""
    Take image and overlay and extract the fluorescence signal of the individual cell objects
"""


def analyze_fluorescence(image, overlay):
    datapoints = []

    # create image (for masks)
    height, width = image.shape[:2]
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    # iterate all rois in the overlay
    for roi in overlay:
        if len(roi) < 3:
            # have at least three coordinates
            continue
        # clear mask image
        draw.rectangle((0, 0, width, height), fill=(0,))

        # draw cell mask
        draw.polygon(tuple(map(lambda coord: tuple(coord), roi)), outline=1, fill=1)
        roi_mask = np.array(img, np.bool)

        # create cell mask for image
        mask = np.broadcast_to(~roi_mask[:, :, None], (*roi_mask.shape, 3))

        # create masked array
        masked_cell = ma.masked_array(image, mask=mask)

        # compute average fluorescence responses
        gray_values = masked_cell[:, :, 0].compressed() / 255
        red_values = masked_cell[:, :, 1].compressed() / 255
        green_values = masked_cell[:, :, 2].compressed() / 255
        average_red = np.median(red_values)
        average_green = np.median(green_values)

        # store the extracted data
        datapoints.append(
            (
                average_red,
                average_green,
            )
        )

    # return extracted data per point
    return datapoints


def render_segmentation_video(video_file, image_and_contours, height, width):
    cell_index = 0

    ve = cv2.VideoWriter(
        video_file, cv2.VideoWriter_fourcc(*"MP4V"), 3, (width, height)
    )
    for i, (image, overlay) in enumerate(tqdm.tqdm(image_and_contours)):
        # draw all cell countours with their respective cluster color
        pil_image = Image.fromarray(
            np.broadcast_to(image[:, :, 0][:, :, None], (*image.shape[:2], 3)), "RGB"
        )
        # overlay.draw(pil_image)
        draw = ImageDraw.Draw(pil_image)
        for roi in overlay:
            draw.polygon(
                tuple(map(lambda coord: tuple(coord), roi)), outline="yellow", fill=None
            )
            # draw.polygon(roi, outline='red', fill=None)
            cell_index += 1

        # convert to raw image
        raw_image = np.asarray(pil_image)

        # convert to bgr (for opencv output)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # add frame to video
        ve.write(raw_image)

    ve.release()


def render_color_clustered_video(video_file, image_and_contours, df, height, width):
    cell_index = 0

    ve = cv2.VideoWriter(
        video_file, cv2.VideoWriter_fourcc(*"MP4V"), 3, (width, height)
    )
    for i, (image, overlay) in enumerate(tqdm.tqdm(image_and_contours)):
        # draw all cell countours with their respective cluster color
        gray_image = np.broadcast_to(image[:, :, 0][:, :, None], (*image.shape[:2], 3))
        # red image
        red_image = np.zeros(gray_image.shape, dtype=np.uint8)
        red_image[:, :, 0] = image[:, :, 1]
        # green image
        green_image = np.zeros(gray_image.shape, dtype=np.uint8)
        green_image[:, :, 1] = image[:, :, 2]

        color_image = red_image + green_image

        res_image = cv2.addWeighted(gray_image, 0.2, color_image, 0.8, 0.0)
        pil_image = Image.fromarray(res_image, "RGB")
        # overlay.draw(pil_image)
        draw = ImageDraw.Draw(pil_image)
        for roi in overlay:
            c = df["color"][cell_index]
            draw.polygon(
                tuple(map(lambda coord: tuple(coord), roi)), outline=c, fill=None
            )
            cell_index += 1

        # convert to raw image
        raw_image = np.asarray(pil_image)

        # convert to bgr (for opencv output)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # add frame to video
        ve.write(raw_image)

    ve.release()


def plot_counts(output_file, df):
    # plot the counts

    red_counts = []
    green_counts = []

    for frame in sorted(np.unique(df["frame"])):
        red_counts.append(len(df[(df.frame == frame) & (df.color == "red")]))
        green_counts.append(len(df[(df.frame == frame) & (df.color == "green")]))

    fig, ax = plt.subplots()
    plt.plot(red_counts, color="red", label="red")
    plt.plot(green_counts, color="green", label="green")

    plt.xlabel("Frame")
    plt.ylabel("Cell count")
    plt.legend()

    plt.savefig(output_file)

    plt.close("all")

    return red_counts, green_counts


def plot_ratio(output_file, red_counts, green_counts):
    fig, ax = plt.subplots()

    ratio = np.array(green_counts) / (np.array(red_counts) + np.array(green_counts))
    ax.plot(ratio, color="blue", label="ratio")
    ax.set_ylabel(r"$\frac{green}{red+green}$")
    ax.set_xlabel("Frame")

    fig.tight_layout(pad=1)

    fig.savefig(output_file)

    plt.close(fig)
