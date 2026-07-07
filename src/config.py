import os

# Paths (change base path if your project in different location)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd()))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FFPP = os.path.join(DATA_DIR, "raw", "ffpp_c23")
RAW_CELEBDF = os.path.join(DATA_DIR, "raw", "celebdf_v2_sampled")

INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
FRAMES_DIR = os.path.join(INTERMEDIATE_DIR, "frames")
ALIGNED_DIR = os.path.join(INTERMEDIATE_DIR, "aligned_faces")
SEQS_DIR = os.path.join(INTERMEDIATE_DIR, "sequences")

# Preprocessing params
FPS = 5                # frames per second to extract
SEQ_LEN = 30           # T = 30
INPUT_SIZE = 260       # CNN input H/W
MIN_FRAMES = 40        # skip videos shorter than this (optional)
NUM_WORKERS = 4

# output formats
SEQ_SAVE_FORMAT = "npy"    # or 'tfrecord'
