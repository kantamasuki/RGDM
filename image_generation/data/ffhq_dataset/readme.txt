The file "ffhq64.tar.gz" is the gzip file of FFHQ images resized to 64x64.

In creating the resized images, I first downloaded the thumbnails128x128 images from the official FFHQ dataset page (https://github.com/NVlabs/ffhq-dataset).
Then, I resized these images by "resize_ffhq.py" with the command "python resize_ffhq.py --img_size 64", which creates the folder "ffhq64"

On behalf of the authors,
Kanta Masuki
