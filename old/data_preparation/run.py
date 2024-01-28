from old.data_preparation.ImageProcessor import ImageProcessor
from old.data_preparation.ImageResizer import ImageResizer


def resize_dataset_image(processed_path):
    resizer = ImageResizer(input_folder='../input/images', output_folder=processed_path)
    resizer.resize_images(target_size=(299, 299))


def process_and_fe(processed_path):
    image_processor = ImageProcessor(processed_path)
    image_processor.process_images()


if __name__ == '__main__':
    resize_dataset_image(processed_path='../processed/images')
    process_and_fe(processed_path='../processed/images')
