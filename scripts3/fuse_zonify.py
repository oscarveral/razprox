from pathlib import Path
import pandas as pd

# Fuses 2 csv files into one
RESOURCES = Path(__file__).parent.parent / "resources3"
INPUT1 = RESOURCES / "zonify_comp" / "zonify_results.csv"
INPUT2 = RESOURCES / "zonify_incomp" / "zonify_results.csv"
OUTPUT = RESOURCES / "zonify_fused" / "zonify_fused_results.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df1 = pd.read_csv(INPUT1)
df2 = pd.read_csv(INPUT2)

df_fused = pd.concat([df1, df2], ignore_index=True)
df_fused.to_csv(OUTPUT, index=False)