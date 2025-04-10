import os

from PIL import Image

#
source_dir = '/all_cells_linear_regression/first_set_of_time_windowed_cells'
dest_dir = '/all_cells_linear_regression/first_set_of_time_windowed_cells/regression_plots/stitched'


## Stitching Images


def stitch_images(image_files, output_path):
    images = [Image.open(file) for file in image_files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.width

    new_image.save(output_path)


# Organize files by the substring after the first underscore
images_to_stitch = {}
for filename in os.listdir(source_dir):
    if filename.endswith('.png'):  # Ensure dealing with PNG files
        identifier = '_'.join(filename.split('_')[1:])
        if identifier in images_to_stitch:
            images_to_stitch[identifier].append(os.path.join(source_dir, filename))
        else:
            images_to_stitch[identifier] = [os.path.join(source_dir, filename)]

# Process each group of three matching images
for key, files in images_to_stitch.items():
    if len(files) == 3:
        # Sort files to maintain consistent order (optional)
        files.sort()
        # Define the output path for the stitched image
        output_path = os.path.join(dest_dir, f"stitched_{key}")
        # Stitch the images
        stitch_images(files, output_path)
        print(f"Stitched images saved to {output_path}")
