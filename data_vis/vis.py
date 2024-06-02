import os
import cv2
import matplotlib.pyplot as plt


def visualize_annotations(image_dir, label_dir, class_names, num_samples=5):
    image_files = os.listdir(image_dir)
    for image_file in image_files[:num_samples]:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(
            label_dir, os.path.splitext(image_file)[0] + '.txt')

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, bbox_width, bbox_height = map(
                    float, line.split())
                x_center, y_center, bbox_width, bbox_height = x_center * \
                    width, y_center * height, bbox_width * width, bbox_height * height
                x_min, y_min = int(x_center - bbox_width /
                                   2), int(y_center - bbox_height / 2)
                x_max, y_max = int(x_center + bbox_width /
                                   2), int(y_center + bbox_height / 2)

                cv2.rectangle(image, (x_min, y_min),
                              (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(image, class_names[int(
                    class_id)], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


def main():
    data_yaml = '../dataset/data.yaml'
    data = load_yaml(data_yaml)
    class_names = data['names']

    for split in ['train', 'val']:
        image_dir = data[split]
        label_dir = os.path.join('dataset/labels', split)

        print(f"Visualizing {split} set:")
        visualize_annotations(image_dir, label_dir, class_names)


if __name__ == '__main__':
    main()
