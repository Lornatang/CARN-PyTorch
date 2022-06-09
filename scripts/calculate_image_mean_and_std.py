# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def main(args):
    train_datasets = ImageFolder(root=args.dataroot, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    image_number = len(train_dataloader)

    for index, (image, _) in enumerate(train_dataloader):
        print(f"Process [{index:08d}/{image_number:08d}] images.")
        for channel in range(3):
            mean[channel] += image[:, channel, :, :].mean()
            std[channel] += image[:, channel, :, :].std()

    mean.div_(len(train_datasets))
    std.div_(len(train_datasets))

    mean_value = list(mean.numpy())
    std_value = list(std.numpy())

    print(f"Mean = [{mean_value[0]:.4f}, {mean_value[1]:.4f}, {mean_value[2]:.4f}]")
    print(f"Std  = [{std_value[0]:.4f}, {std_value[1]:.4f}, {std_value[2]:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the mean and std of a dataset.")
    parser.add_argument("--dataroot", type=str, help="Dataset root directory path.")
    args = parser.parse_args()

    main(args)
