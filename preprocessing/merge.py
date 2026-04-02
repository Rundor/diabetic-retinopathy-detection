import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# =======================
# PATHS (EDIT THESE)
# =======================
APTOS_PATH = 'Dataset3_Final_Project'
FP_PATH = '29423747\MMRDR.zip\MMRDR-CFP'
OUTPUT_PATH = 'merged_dataset'

APTOS_IMG = os.path.join(APTOS_PATH, "all_images")
FP_IMG = os.path.join(FP_PATH, "img")

# =======================
# LOAD APTOS
# =======================
train_df = pd.read_csv(os.path.join(APTOS_PATH, "aptos_train_70_20_10.csv"))
val_df   = pd.read_csv(os.path.join(APTOS_PATH, "aptos_val_70_20_10.csv"))
test_df  = pd.read_csv(os.path.join(APTOS_PATH, "aptos_test_70_20_10.csv"))

aptos_df = pd.concat([train_df, val_df, test_df])

# =======================
# LOAD FP
# =======================
fp_df = pd.read_csv(os.path.join(FP_PATH, "FP.csv"))
fp_df = fp_df[['id_code', 'diagnosis']]

# =======================
# CLASS DISTRIBUTION
# =======================
aptos_counts = aptos_df['diagnosis'].value_counts().to_dict()
fp_counts = fp_df['diagnosis'].value_counts().to_dict()

print("APTOS:", aptos_counts)
print("FP:", fp_counts)

# =======================
# TARGET BALANCE
# =======================
all_classes = [0,1,2,3,4]
target = max(aptos_counts.values())

print("Target per class:", target)

# =======================
# SAMPLE FROM FP
# =======================
selected_fp = []

for cls in all_classes:
    aptos_n = aptos_counts.get(cls, 0)
    needed = max(0, target - aptos_n)

    fp_class = fp_df[fp_df['diagnosis'] == cls]

    if len(fp_class) == 0:
        continue

    sampled = fp_class.sample(
        n=min(needed, len(fp_class)),
        random_state=42
    )

    selected_fp.append(sampled)

fp_selected_df = pd.concat(selected_fp)

# =======================
# SPLIT FP (same ratios as APTOS)
# =======================
train_ratio = len(train_df) / len(aptos_df)
val_ratio   = len(val_df)   / len(aptos_df)
test_ratio  = len(test_df)  / len(aptos_df)

train_fp, temp_fp = train_test_split(
    fp_selected_df,
    test_size=(1 - train_ratio),
    stratify=fp_selected_df['diagnosis'],
    random_state=42
)

val_size_adjusted = val_ratio / (val_ratio + test_ratio)

val_fp, test_fp = train_test_split(
    temp_fp,
    test_size=(1 - val_size_adjusted),
    stratify=temp_fp['diagnosis'],
    random_state=42
)

# =======================
# FORMAT FP DATA
# =======================
def format_fp(df):
    df = df.copy()
    
    # extract only filename (remove 'img/')
    df['id_code'] = df['id_code'].apply(lambda x: os.path.basename(x))
    
    # add prefix
    df['id_code'] = df['id_code'].apply(lambda x: "fp_" + x)
    
    return df

train_fp = format_fp(train_fp)
val_fp   = format_fp(val_fp)
test_fp  = format_fp(test_fp)

# =======================
# MERGE CSVs
# =======================
new_train = pd.concat([train_df, train_fp])
new_val   = pd.concat([val_df, val_fp])
new_test  = pd.concat([test_df, test_fp])

# =======================
# CREATE OUTPUT FOLDERS
# =======================
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "all_images"), exist_ok=True)

OUT_IMG = os.path.join(OUTPUT_PATH, "all_images")

# =======================
# COPY APTOS IMAGES
# =======================
print("Copying APTOS images...")
for fname in os.listdir(APTOS_IMG):
    src = os.path.join(APTOS_IMG, fname)
    dst = os.path.join(OUT_IMG, fname)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

# =======================
# COPY FP IMAGES
# =======================
def copy_fp_images(df):
    for _, row in df.iterrows():
        # remove prefix
        filename = row['id_code'].replace("fp_", "")
        
        # original path (WITH img/)
        src = os.path.join(FP_IMG, filename)

        if os.path.exists(src):
            dst = os.path.join(OUT_IMG, row['id_code'])
            shutil.copy(src, dst)
        else:
            print("Missing:", src)
            
print("Copying FP images...")
copy_fp_images(train_fp)
copy_fp_images(val_fp)
copy_fp_images(test_fp)

# =======================
# SAVE CSVs
# =======================
new_train.to_csv(os.path.join(OUTPUT_PATH, "train.csv"), index=False)
new_val.to_csv(os.path.join(OUTPUT_PATH, "val.csv"), index=False)
new_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False)

print("✅ Done! Merged dataset created at:", OUTPUT_PATH)
