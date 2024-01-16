import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os
import logging
from tqdm import tqdm
from multiprocessing import Pool
from LDR import *
from tone import *
from genStroke_origin import *
from drawpatch import rotate
from tools import *
from ETF.edge_tangent_flow import *
from deblue import deblue
from quicksort import *
import sys
import argparse

np.random.seed(1)

N = 10  # Quantization order
PERIOD = 5  # line period
DIRECTION = 10  # num of dir
FREQ = 10  # save every（freq) lines drawn
DEEPEN = 1  # for edge
TRANSTONE = False  # for Tone8
KERNEL_RADIUS = 3  # for ETF
ITER_TIME = 15  # for ETF
BACKGROUND_DIR = None  # for ETF
CLAHE = True
EDGE_CLAHE = True
DRAW_NEW = True
RANDOM_ORDER = False
ETF_ORDER = True
PROCESS_VISIBLE = True
NUM_PROCESSES = 4

logger = logging.getLogger("skechify_images")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"sketchify_images.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def setup_dirs(input_path, output_path):
    file_name = os.path.basename(input_path)
    file_name = file_name.split(".")[0]
    logging.info(f"Creating Sketch for: {file_name}")
    output_path = output_path + "/" + file_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + "/mask")
        os.makedirs(output_path + "/process")
    else:
        raise Exception("Output directory already exists")
    return output_path


def apply_etf_filter(
    input_path,
    output_path,
):
    start_time = time.time()
    ETF_filter = ETF(
        input_path=input_path,
        output_path=output_path + "/mask",
        dir_num=DIRECTION,
        kernel_radius=KERNEL_RADIUS,
        iter_time=ITER_TIME,
        background_dir=BACKGROUND_DIR,
    )
    ETF_filter.forward()

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60.0
    logger.info(f"ETF completed ... {duration_minutes:.2f} minutes")


def create_grayscale_image(input_path, output_path):
    start_time = time.time()
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    (h0, w0) = input_img.shape

    cv2.imwrite(output_path + "/input_gray.jpg", input_img)

    if TRANSTONE == True:
        input_img = transferTone(input_img)

    now_ = np.uint8(np.ones((h0, w0))) * 255

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60.0
    logger.info(f"Creating grayscale image ... {duration_minutes:.2f} minutes")

    return input_img, now_, h0, w0


def process_directions(output_path):
    stroke_sequence = []
    for dirs in tqdm(range(DIRECTION), desc="Processing directions: "):
        stroke_temp = {
            "angle": None,
            "grayscale": None,
            "row": None,
            "begin": None,
            "end": None,
        }
        angle = -90 + dirs * 180 / DIRECTION
        stroke_temp["angle"] = angle
        img, _ = rotate(input_img, -angle)

        if CLAHE == True:
            img = HistogramEqualization(img)

        img_pad = cv2.copyMakeBorder(
            img,
            2 * PERIOD,
            2 * PERIOD,
            2 * PERIOD,
            2 * PERIOD,
            cv2.BORDER_REPLICATE,
        )
        img_normal = cv2.normalize(
            img_pad.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX
        )

        x_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 1, 0, ksize=5)
        y_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 0, 1, ksize=5)

        x_der = torch.from_numpy(x_der) + 1e-12
        y_der = torch.from_numpy(y_der) + 1e-12

        gradient_magnitude = torch.sqrt(x_der**2.0 + y_der**2.0)
        gradient_norm = gradient_magnitude / gradient_magnitude.max()

        ldr = LDR(img, N)
        cv2.imwrite(output_path + "/Quantization.png", ldr)

        LDR_single_add(ldr, N, output_path)

        (h, w) = ldr.shape
        canvas = Gassian((h + 4 * PERIOD, w + 4 * PERIOD), mean=250, var=3)

        for j in range(N):
            stroke_temp["grayscale"] = j * 256 / N
            mask = (
                cv2.imread(
                    output_path + "/mask/mask{}.png".format(j), cv2.IMREAD_GRAYSCALE
                )
                / 255
            )
            dir_mask = cv2.imread(
                output_path + "/mask/dir_mask{}.png".format(dirs),
                cv2.IMREAD_GRAYSCALE,
            )

            dir_mask, _ = rotate(dir_mask, -angle, pad_color=0)
            dir_mask[dir_mask < 128] = 0
            dir_mask[dir_mask > 127] = 1

            distensce = Gassian((1, int(h / PERIOD) + 4), mean=PERIOD, var=1)
            distensce = np.uint8(
                np.round(np.clip(distensce, PERIOD * 0.8, PERIOD * 1.25))
            )
            raw = -int(PERIOD / 2)

            for i in np.squeeze(distensce).tolist():
                if raw < h:
                    y = raw + 2 * PERIOD
                    raw += i
                    for interval in get_start_end(
                        mask[y - 2 * PERIOD] * dir_mask[y - 2 * PERIOD]
                    ):
                        begin = interval[0]
                        end = interval[1]

                        begin -= 2 * PERIOD
                        end += 2 * PERIOD

                        length = end - begin
                        stroke_temp["begin"] = begin
                        stroke_temp["end"] = end
                        stroke_temp["row"] = y - int(PERIOD / 2)

                        stroke_temp["importance"] = (
                            255 - stroke_temp["grayscale"]
                        ) * torch.sum(
                            gradient_norm[
                                y : y + PERIOD,
                                interval[0] + 2 * PERIOD : interval[1] + 2 * PERIOD,
                            ]
                        ).numpy()

                        stroke_sequence.append(stroke_temp.copy())
    return stroke_sequence, h, w


def process_stroke_chunk(stoke_chunk, process_id, canvases, h0, w0, w, output_path):
    result = Gassian((h0, w0), mean=250, var=3)
    step = 0
    Freq = FREQ
    for stroke_temp in tqdm(stoke_chunk, desc=f"Drawing strokes {process_id}: "):
        angle = stroke_temp["angle"]
        dirs = int((angle + 90) * DIRECTION / 180)
        grayscale = stroke_temp["grayscale"]
        distribution = ChooseDistribution(period=PERIOD, Grayscale=grayscale)
        row = stroke_temp["row"]
        begin = stroke_temp["begin"]
        end = stroke_temp["end"]
        length = end - begin

        newline = Getline(distribution=distribution, length=length)

        canvas = canvases[dirs]

        if length < 1000 or begin == -2 * PERIOD or end == w - 1 + 2 * PERIOD:
            temp = canvas[row : row + 2 * PERIOD, 2 * PERIOD + begin : 2 * PERIOD + end]
            m = np.minimum(temp, newline[:, : temp.shape[1]])
            canvas[row : row + 2 * PERIOD, 2 * PERIOD + begin : 2 * PERIOD + end] = m

        now, _ = rotate(
            canvas[2 * PERIOD : -2 * PERIOD, 2 * PERIOD : -2 * PERIOD], angle
        )
        (H, W) = now.shape
        now = now[
            int((H - h0) / 2) : int((H - h0) / 2) + h0,
            int((W - w0) / 2) : int((W - w0) / 2) + w0,
        ]
        result = np.minimum(now, result)
        if PROCESS_VISIBLE == True:
            cv2.imshow('step', result)
            cv2.waitKey(1)
            
        step += 1

        if step % Freq == 0:
            cv2.imwrite(output_path + "/process/{0:04d}.jpg".format(int(step/Freq)) , result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketchify Images.")
    parser.add_argument("--input_path", type=str, help="path to the input image")
    parser.add_argument("--output_path", type=str, help="path to the output directory")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    output_path = setup_dirs(input_path, output_path)
    apply_etf_filter(input_path, output_path)

    input_img, now_, h0, w0 = create_grayscale_image(input_path, output_path)

    step = 0
    if DRAW_NEW == True:
        stroke_sequence, h, w = process_directions(output_path)

        if RANDOM_ORDER == True:
            random.shuffle(stroke_sequence)

        if ETF_ORDER == True:
            random.shuffle(stroke_sequence)
            quickSort(stroke_sequence, 0, len(stroke_sequence) - 1)
        result = Gassian((h0, w0), mean=250, var=3)
        canvases = []

        for dirs in range(DIRECTION):
            angle = -90 + dirs * 180 / DIRECTION
            canvas, _ = rotate(result, -angle)
            canvas = np.pad(
                canvas,
                pad_width=2 * PERIOD,
                mode="constant",
                constant_values=(255, 255),
            )
            canvases.append(canvas)

        chunks = np.array_split(stroke_sequence, NUM_PROCESSES)
        args = [(chunk, i, canvases, h0, w0, w, output_path) for i, chunk in enumerate(chunks)]
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.starmap(process_stroke_chunk, args)

        stacked_images = np.stack(results, axis=-1)
        result = np.min(stacked_images, axis=-1)

        if step % FREQ != 0:
            step = int(step / FREQ) + 1
            cv2.imwrite(output_path + "/process/{0:04d}.jpg".format(step), result)

        cv2.destroyAllWindows()
        cv2.imwrite(output_path + "/draw.jpg", result)

    edge = genStroke(input_img, 18)
    edge = np.power(edge, DEEPEN)
    edge = np.uint8(edge * 255)
    if EDGE_CLAHE == True:
        edge = HistogramEqualization(edge)

    cv2.imwrite(output_path + "/edge.jpg", edge)
    cv2.imshow("edge", edge)

    edge = np.float32(edge)
    now_ = cv2.imread(output_path + "/draw.jpg", cv2.IMREAD_GRAYSCALE)
    result = res_cross = np.float32(now_)

    result[1:, 1:] = np.uint8(edge[:-1, :-1] * res_cross[1:, 1:] / 255)
    result[0] = np.uint8(edge[0] * res_cross[0] / 255)
    result[:, 0] = np.uint8(edge[:, 0] * res_cross[:, 0] / 255)
    result = edge * res_cross / 255
    result = np.uint8(result)

    cv2.imwrite(output_path + "/result.jpg", result)
    cv2.imshow("result", result)

    deblue(result, output_path)

    img_rgb_original = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path + "/input.jpg", img_rgb_original)
    img_yuv = cv2.cvtColor(img_rgb_original, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = result
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imwrite(output_path + "/result_RGB.jpg", img_rgb)
