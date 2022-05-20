import matplotlib

matplotlib.use("Agg")
import tifffile
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import tqdm


from base import (
    pairwise_distances,
    plot_counts,
    plot_ratio,
    render_color_clustered_video,
    segmentation,
    extract_fluorescence,
    render_segmentation_video,
)

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform CoNos Analysis")
    parser.add_argument(
        "input", type=str, nargs=1, help="Image stack for the analysis (e.g. input.tif)"
    )

    args = parser.parse_args()

    from base import (
        plot_counts,
        plot_ratio,
        render_color_clustered_video,
        segmentation,
        extract_fluorescence,
        render_segmentation_video,
    )

    import argparse

    # Read images
    images = tifffile.imread(args.input)

    print(images[0].dtype)

    # Turn to graysacle
    gray_images = list(map(lambda image: image[:, :, 0], images))

    # perform segmentation
    predictions = segmentation(gray_images, True)

    # filter segmentation
    from shapely.geometry import Polygon
    import tqdm

    areas = []
    filtered_predictions = []

    # parameters for filtering
    min_area = 0.5
    max_area = 4.5
    max_length_width_ratio = 5.5

    # filter predictions for every frame
    for frame, (image, overlay) in enumerate(tqdm.tqdm(zip(images, predictions))):
        filtered_overlay = []
        for det in overlay:
            p = Polygon(det)
            length = (
                np.max(
                    pairwise_distances(
                        np.array(p.minimum_rotated_rectangle.exterior.coords)
                    )
                )
                * 0.07
            )
            width = (
                np.min(
                    pairwise_distances(
                        np.array(p.minimum_rotated_rectangle.exterior.coords)
                    )
                )
                * 0.07
            )

            area = p.area * 0.07 ** 2  # area in microns^2

            areas.append(area)

            if min_area < area and max_area > area:
                if length / width < max_length_width_ratio:
                    filtered_overlay.append(det)

        filtered_predictions.append(filtered_overlay)

    raw_prediction = predictions
    predictions = filtered_predictions

    # render segmentation video
    height, width = images[0].shape[:2]
    render_segmentation_video(
        "output/segmentation.avi", zip(images, filtered_predictions), height, width
    )

    filtered_images = []

    for image in tqdm.tqdm(images):
        filtered_channels = []
        for channel in range(image.shape[2]):
            filtered_channel = gaussian_filter(image[:, :, channel], sigma=5)
            filtered_channels.append(filtered_channel)
        filtered_images.append(np.stack(filtered_channels, axis=-1))
        # print(filtered_images[-1].shape)

    # extract fluorescence
    df = extract_fluorescence(filtered_images, filtered_predictions)

    def normalize(values, min=None, max=None):
        if min is None:
            min = np.min(values)
        if max is None:
            max = np.max(values)
        return (values - min) / (max - min)

    df_normalized = df.copy(True)

    for frame, image in enumerate(filtered_images):
        frame_df = df[df["frame"] == frame]
        df_normalized.loc[df_normalized["frame"] == frame, "red"] = normalize(
            frame_df["red"]
        )  # , np.min(image[:,:,1]/255.), np.max(image[:,:,1]/255.))
        df_normalized.loc[df_normalized["frame"] == frame, "green"] = normalize(
            frame_df["green"]
        )  # , np.min(image[:,:,2]/255.), np.max(image[:,:,2]/255.))

    plt.hist(df_normalized["red"])
    plt.hist(df_normalized["green"])

    print(np.max(df_normalized["red"]))

    # df_copy = df.copy(True)
    df = df_normalized

    # perform clustering
    color = np.array(["yellow", "green", "red"])

    prediction = np.zeros((len(df),), dtype=int)  # by default yellow label

    num_frames = np.max(df["frame"])
    all_thresholds = []

    fluorescence_threshold = 0.35

    for i, row in df.iterrows():
        # threshold clustering based on fluorescence
        if (
            row["red"] >= fluorescence_threshold
            and row["green"] >= fluorescence_threshold
        ):
            if row["red"] >= row["green"]:
                prediction[i] = 2  # red label
            else:
                prediction[i] = 1  # green label
        elif row["red"] >= fluorescence_threshold:
            prediction[i] = 2  # red label
        elif row["green"] >= fluorescence_threshold:
            prediction[i] = 1  # green label

    # store the color
    df["color"] = color[prediction]

    # show clustering
    # scatter data with correct colors
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(df["red"], df["green"], c=color[prediction], s=50, alpha=0.01)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.savefig("output/clustering.png")

    plt.close(fig)

    # make clustering video
    render_color_clustered_video(
        "output/clustered.avi", zip(images, filtered_predictions), df, height, width
    )

    from base import plot_ratio

    # produce plots
    red_counts, green_counts = plot_counts("output/counts.png", df)

    plt.close("all")
    plot_ratio("output/ratio.png", red_counts, green_counts)

    # output counts as csv
    import csv

    with open("output/counts.csv", "w", newline="\n") as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        spamwriter.writerow(["Frame", "Red count", "Green Count"])
        for frame in range(len(red_counts)):
            spamwriter.writerow([frame, red_counts[frame], green_counts[frame]])
