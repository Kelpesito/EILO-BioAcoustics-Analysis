import time
from tqdm.auto import tqdm

n_folds=5

for fold in (cv_bar := tqdm(range(1, n_folds + 1), desc="CV", unit="fold", leave=True)):
    tqdm.write("=" * 60)
    tqdm.write(f"FOLD {fold}")
    tqdm.write("=" * 60)       
    for i in tqdm(range(3), desc="uwu", leave=True):
        tqdm.write("Performing things...")
        time.sleep(3)
    tqdm.write("Results: XD")