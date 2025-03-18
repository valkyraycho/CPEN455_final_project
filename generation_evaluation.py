"""
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify other code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 20-33)
2. Modify how you call your sample function(line 50-55)

REQUIREMENTS:
- You should save the generated images to the gen_data_dir, which is fixed as './samples'
- If you directly run this code, it should generate images and calculate the FID score, you should follow the same format as the demonstration, there should be 100 images in 4 classes, each class has 25 images
"""

import argparse
import os

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

from dataset import *
from model import *
from utils import *

# TODO: Begin of your code
# This is a demonstration of how to call the sample function, feel free to modify it
# You should modify this sample function to get the generated images from your model
# You should save the generated images to the gen_data_dir, which is fixed as 'samples'
sample_op = lambda x: sample_from_discretized_mix_logistic(x, 5)


def my_sample(
    model, gen_data_dir, sample_batch_size=25, obs=(3, 32, 32), sample_op=sample_op
):
    device = next(model.parameters()).device
    for label_name, label in my_bidict.items():
        print(f"Label: {label}")
        # generate images for each label, each label has 25 images
        sample_t = sample(
            model,
            sample_batch_size,
            obs,
            sample_op,
            labels=torch.full(
                (sample_batch_size,), label, dtype=torch.long, device=device
            ),
        )
        sample_t = rescaling_inv(sample_t)
        save_images(sample_t, os.path.join(gen_data_dir), label=label_name)


# End of your code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--ref_data_dir",
        type=str,
        default="data/test",
        help="Location for the dataset",
    )

    args = parser.parse_args()

    ref_data_dir = args.ref_data_dir
    gen_data_dir = os.path.join(os.path.dirname(__file__), "samples")
    BATCH_SIZE = 128
    device = get_device()

    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

    # TODO: Begin of your code
    # Load your model and generate images in the gen_data_dir, feel free to modify the model
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5)
    model = model.to(device)
    model.load_state_dict(
        torch.load("models/conditional_pixelcnn.pth", map_location=device)
    )
    model = model.eval()
    # End of your code

    my_sample(model=model, gen_data_dir=gen_data_dir)

    paths = [gen_data_dir, ref_data_dir]
    print(
        f"#generated images: {len(os.listdir(gen_data_dir)):d}, #reference images: {len(os.listdir(ref_data_dir)):d}"
    )

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print(f"Dimension {192:d} works! fid score: {fid_score}")
    except:
        print(f"Dimension {192:d} fails!")

    print(f"Average fid score: {fid_score}")
