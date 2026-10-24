# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
from image_baseline_profiler_core import main

if __name__ == "__main__":
    main(default_architecture="ahfunet")
