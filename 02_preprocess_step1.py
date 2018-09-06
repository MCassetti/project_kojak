import os
from PIL import Image

def resize_image(image,size):
    return image.resize(size, Image.ANTIALIAS)

def process_and_save(image_dir, output_dir, size):
    """Resize the images in image_dir and save into output_dir"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        #with open(os.path.join(image_dir, image), 'r+b') as f:
            image_path = os.path.join(image_dir, image)
            #img = Image(image_path)
            img = Image.open(image_path)
            img = resize_image(img, size)
            img.save(os.path.join(output_dir, image), img.format)


if __name__ == '__main__':
    base_path = os.getcwd()
    meme_path = '/image_downloads/'
    image_string = '/image_resized/'
    size = [256, 256]

    image_dir = base_path + meme_path
    output_dir = base_path + image_string
    process_and_save(image_dir,output_dir,size)
