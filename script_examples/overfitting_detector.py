import argparse
import base64
import io
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, display
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from torchvision import models, transforms


def find_elbow(ks, scores):
    diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

    # plt.plot(ks[:-1], diffs)
    # plt.scatter(ks[:-1], diffs)
    # plt.grid()
    # plt.title("Elbow Point")
    # plt.show()

    # 二乗誤差が最も小さい点のindexを返す
    best_k = np.argmin(np.abs(np.diff(diffs))) + 1
    return ks[best_k]


class OverfittingDetector:
    def __init__(self):
        # VGG16モデルの特徴抽出
        self.model = models.vgg16(pretrained=True)
        self.model.eval()

    def extract_features(self, img_path):
        # 画像の読み込みと前処理
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img = transform(Image.open(img_path)).unsqueeze(0)

        # モデルで特徴量を抽出
        features = self.model(img)

        # 3次元を2次元に平坦化
        flat_features = features.reshape(-1)
        return flat_features.detach().numpy()

    def calculate_diff(self, features1, features2):
        return np.linalg.norm(features1 - features2)

    def detect_overfitting(self, diff_sum, threshold):
        if diff_sum > threshold:
            return True
        else:
            return False

    def check_overfitting(self, input_dir, overfitting_sample_img_path, threshold):
        print("check_overfitting sample=", overfitting_sample_img_path)
        overfitting_sample_features = self.extract_features(overfitting_sample_img_path)
        diff_sum = 0
        img_path_list = []
        for filename in sorted(os.listdir(input_dir), key=str):
            if filename.endswith(".png"):
                img_path = os.path.join(input_dir, filename)
                img_path_list.append(img_path)
                # features = self.extract_features(img_path)
                # diff = self.calculate_diff(features, overfitting_sample_features)
                # print("diff=", diff, "filename=", filename)
                # diff_sum += diff

        cluster_samples_dict = self.check_dataset_variation(img_path_list)
        is_overfitting = False
        # is_overfitting = self.detect_overfitting(diff_sum, threshold)
        return cluster_samples_dict, is_overfitting

    def cluster_images(self, image_paths):
        features = [self.extract_features(path) for path in image_paths]

        # エルボー法で最適なクラスタ数を取得
        ks = range(2, 10)
        scores = [
            calinski_harabasz_score(
                features, KMeans(n_clusters=k).fit(features).labels_
            )
            for k in ks
        ]
        k = find_elbow(ks, scores)

        kmeans = KMeans(n_clusters=k).fit(features)
        return kmeans.labels_

    def check_dataset_variation(self, image_paths):
        labels = self.cluster_images(image_paths)

        # 各クラスタの画像数を表示
        for i in set(labels):
            n_images = len([1 for l in labels if l == i])
            print(f"Cluster {i}: {n_images} images")

        cluster_samples_dict = self.display_cluster_samples(image_paths, labels)
        return cluster_samples_dict

    def display_cluster_samples(self, image_paths, labels):
        clusters = set(labels)
        cluster_samples_dict = {}
        for c in clusters:
            cluster_paths = [p for i, p in enumerate(image_paths) if labels[i] == c]

            if len(cluster_paths) > 5:
                samples = random.sample(cluster_paths, 5)
            else:
                samples = cluster_paths

            imgs = [Image.open(p) for p in samples]
            cluster_samples_dict[c] = imgs
        return cluster_samples_dict

    def save_cluster_samples_html(self, cluster_samples_dict, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        html = ""

        for c, imgs in cluster_samples_dict.items():
            html += f"<h3>Cluster {c}</h3><br>\n"
            html += "<div style='display: flex;'>\n"
            for img in imgs:
                with io.BytesIO() as output:
                    img.save(output, format="PNG")
                    img_bytes = output.getvalue()

                    base64_bytes = base64.b64encode(img_bytes)
                    base64_str = base64_bytes.decode("utf-8")

                    img_tag = (
                        f'<img src="data:image/png;base64,{base64_str}" width="200">'
                    )
                    html += img_tag
            html += "</div>\n"
            html += "<br>\n"

        out_path = os.path.join(out_dir, "clusters.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        # print("html=", html)


def arg_parser():
    parser = argparse.ArgumentParser(description="overfitting_detector.py")
    parser.add_argument("--input_dir", "-i", help="input_dir")
    parser.add_argument("--output_dir", "-o", help="output_dir", default="output/html")
    options = parser.parse_args()
    print("input_dir=" + options.input_dir)
    print("output_dir=" + options.output_dir)
    return options


if __name__ == "__main__":
    options_ = arg_parser()
    input_dir_ = options_.input_dir
    output_dir_ = options_.output_dir
    OVERFITTING_SAMPLE_IMG_PATH = "../output/overfitting_sample/overfitting_sample.png"
    THRESHOLD = 100
    od = OverfittingDetector()
    cluster_samples_dict, is_overfitting = od.check_overfitting(
        input_dir_, OVERFITTING_SAMPLE_IMG_PATH, THRESHOLD
    )
    print(is_overfitting)
    od.save_cluster_samples_html(cluster_samples_dict, output_dir_)
