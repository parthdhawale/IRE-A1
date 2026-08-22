import pandas as pd
import pyarrow as pa
import time
from src.pipeline.preprocess import preprocess_ebnerd

print("Loading...")
# We can't slice inside preprocess_ebnerd easily, so we just mock the exact same types
N = 1_000_000
print(f"Creating mock DF of {N} rows...")
df = pd.DataFrame({
    'impression_id': [str(i) for i in range(N)],
    'user_id': [str(i) for i in range(N)],
    'timestamp': pd.date_range('2023-01-01', periods=N),
    'click_history': [['123', '456'] for _ in range(N)],
    'candidates': [['123', '456', '789'] for _ in range(N)],
    'labels': [[0, 1, 0] for _ in range(N)],
})

print("Calling pa.Schema.from_pandas...")
t0 = time.time()
schema = pa.Schema.from_pandas(df)
print(f"Time: {time.time()-t0:.2f}s")
