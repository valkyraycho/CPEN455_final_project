import argparse
import csv
import os
from pprint import pprint

import torch
from torchvision import transforms
from tqdm import tqdm

from classification_evaluation import get_label
from dataset import *
from model import *
from utils import *


def classify(model, data_loader, device, csv_test_file, csv_output_file_name):
    # Read all img names from ./data/test.csv
    img_names = []
    with open(csv_test_file) as file:
        reader = csv.reader(file)

        # Iterate over each row in the CSV file
        for row in reader:
            # Should ignore the -1 dummy label and also drop the test/ prefix
            img_name = row[0].split(",")[0]

            img_names.append(img_name)

    model.eval()
    img_idx = 0

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        answer = get_label(model, model_input, device)
        for label in answer.cpu().detach().numpy():
            img_names[img_idx] = [img_names[img_idx], str(label)]
            img_idx += 1

    # Prepare CSV for submission
    with open(csv_output_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write classes
        for row in img_names:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--data_dir", type=str, default="data", help="Location for the dataset"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="test", help="Mode for the dataset"
    )

    args = parser.parse_args()
    pprint(args.__dict__)
    device = get_device()
    kwargs = {"num_workers": 0, "pin_memory": True, "drop_last": False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
    )

    # TODO:Begin of your code
    # You should replace the random classifier with your trained model
    # model = random_classifier(NUM_CLASSES)
    model = PixelCNN(nr_resnet=3, nr_filters=160, input_channels=3, nr_logistic_mix=10)
    # End of your code

    model = model.to(device)
    # Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    # You should save your model to this path
    model_path = os.path.join(
        os.path.dirname(__file__), "models/conditional_pixelcnn.pth"
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("model parameters loaded")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()

    classify(
        model=model,
        data_loader=dataloader,
        device=device,
        csv_test_file="./data/test.csv",
        csv_output_file_name="submission.csv",
    )
