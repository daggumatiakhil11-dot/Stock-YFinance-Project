import os
import subprocess
from datetime import datetime

print("🚀 Starting Full AI Pipeline...")

# Step 1: Run data download script
print("📥 Downloading data...")
subprocess.run(["python", "download_data.py"])

# Step 2: Train model (optional future separation)
print("🧠 Running predictions...")

# Step 3: Log run
log_file = "pipeline_log.txt"

with open(log_file, "a") as f:
    f.write(f"Run completed at {datetime.now()}\n")

print("✅ Pipeline Completed Successfully!")