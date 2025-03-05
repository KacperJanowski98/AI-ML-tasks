"""
Configuration settings for the dataset creation process
"""

# Dataset parameters
NUM_SAMPLES_PER_PHRASE = 9  # 3 noise types x 3 SNR levels
SAMPLE_RATE = 16000  # Standard rate for Whisper

# Noise parameters
SNR_LEVELS = [5, 10, 15]  # Signal-to-noise ratios in dB
NOISE_TYPES = ["wind", "traffic", "crowd"]  # Types of background noise

# Paths
NOISE_SAMPLES_DIR = "data/noise_samples"
OUTPUT_DIR = "data/generated"
DATASET_PATH = "data/generated/police_dataset.json"

# TTS parameters
TTS_RATE = 24000  # Default gTTS sampling rate

# Domain-specific phrases
POLICE_PHRASES = [
    "Suspect detained at intersection of Main and Broadway, requesting backup.",
    "Code 10-31 in progress at 1420 Elm Street, proceeding with caution.",
    "Officer requesting 10-78, hostile crowd forming near the plaza.",
    "Vehicle matching BOLO description spotted heading eastbound on Highway 42.",
    "Witness describes perpetrator as male, approximately 6'2\", wearing dark hoodie and jeans.",
    "Proceeding to serve warrant at 728 Oak Drive, requesting additional units.",
    "Dispatch, we have a 10-50 at the corner of 5th and Pine, medical assistance required.",
    "K-9 unit deployed to track suspect heading north through Central Park.",
    "Officer down at 1235 West Boulevard, requesting immediate assistance.",
    "Traffic stop initiated on silver sedan, license plate Alpha-Bravo-Charlie-123.",
    "Surveillance footage shows suspect entering the building at approximately 2300 hours.",
    "Code 10-15, transporting suspect to central booking.",
    "Be advised, suspect is considered armed and dangerous, proceed with caution.",
    "Multiple witnesses confirm disturbance at the south entrance of the mall.",
    "Stolen vehicle recovered at abandoned warehouse on Industrial Drive."
]
