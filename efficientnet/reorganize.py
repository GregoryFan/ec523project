import os
import pandas as pd
import shutil

base = "../../Composite data/Composite data/Testing"
csv_path = os.path.join(base, "Testing_labels.csv")

# CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    img_name = row["image_name"]
    label = row["label"]

    src = os.path.join(base, img_name)
    dst_dir = os.path.join(base, label)

    os.makedirs(dst_dir, exist_ok=True) #make directory if doesn't exist

    dst = os.path.join(dst_dir, img_name)

    #move image if exists
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"Warning: {src} not found!")

print("Dataset reorganized successfully!")
