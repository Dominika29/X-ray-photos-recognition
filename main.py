import torch
from pathlib import Path
import os

torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


CLASS_NAMES = ["fractured", "not fractured"]


fracatlas_root_dir = Path("./FracAtlas")
fracatlas_train_dir = Path(f"{fracatlas_root_dir}/train")
fracatlas_test_dir = Path(f"{fracatlas_root_dir}/test")
fracatlas_val_dir = Path(f"{fracatlas_root_dir}/val")
fracatlas_non_fractured_dir = Path(f"{fracatlas_root_dir}/not fractured")

output_dir = Path("./output")


train_image_filepaths = [x for x in os.listdir(f"{fracatlas_train_dir}/img")]
train_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_train_dir}/ann")]

test_image_filepaths = [x for x in os.listdir(f"{fracatlas_test_dir}/img")]
test_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_test_dir}/ann")]

train_image_filepaths.extend(test_image_filepaths)

# train + test combined datasets are all fractured
fractured_image_filepaths = train_image_filepaths

non_fractured_image_filepaths = [x for x in os.listdir(f"{fracatlas_non_fractured_dir}/img")]
non_fractured_annotation_filepaths = [x for x in os.listdir(f"{fracatlas_non_fractured_dir}/ann")]



for c in CLASS_NAMES:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)


def main():
    print(f"count of fractured images: {len(fractured_image_filepaths)}")
    print(f"count of non fractured images: {len(non_fractured_image_filepaths)}")


if __name__ == "__main__":
    main()