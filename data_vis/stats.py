import os
import yaml
from collections import defaultdict


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def count_annotations(label_dir):
    counts = defaultdict(int)
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    counts[class_id] += 1
    return counts


def main():
    data_yaml = 'dataset/data.yaml'
    data = load_yaml(data_yaml)

    for split in ['train', 'val', 'test']:
        image_dir = data[split]
        label_dir = os.path.join('dataset/labels', split)

        num_images = len(os.listdir(image_dir))
        annotation_counts = count_annotations(label_dir)

        print(f"Statistics for {split} set:")
        print(f"Number of images: {num_images}")
        print(f"Annotations per class: {dict(annotation_counts)}")
        print()


if __name__ == '__main__':
    main()
