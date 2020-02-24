import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path


def main(voxceleb_path="D:\\VoxCeleb/vox1_dev_wav/wav"):
    dest = os.path.join(voxceleb_path, "**/*.wav")
    print(dest)
    flist = glob.glob(dest, recursive=True)
    short_path = [f.replace(f"{voxceleb_path}\\", "") for f in flist]
    files_df = pd.DataFrame({"dest": flist, "short_path": short_path})
    files_df["id"] = files_df["short_path"].apply(lambda x: x.split("\\")[0])
    files_df["link"] = files_df["short_path"].apply(lambda x: x.split("\\")[1])
    files_df["class"] = pd.to_numeric(files_df["id"].str.replace("id", "")) - 10001
    train, test = train_test_split(files_df, test_size=0.09, stratify=files_df["class"])
    train.sort_values("class", inplace=True)
    test.sort_values("class", inplace=True)
    train[["short_path", "class"]].to_csv(
        "../data/processed/voxlb1_train.txt", header=False, index=False, sep=" "
    )
    test[["short_path", "class"]].to_csv(
        "../data/processed/voxlb1_val.txt", header=False, index=False, sep=" "
    )


main()
