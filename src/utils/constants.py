DEBUG = False

MAX_LEVEL = 15
RESAMPLE_LOW_FREQUENCY_CATEGORIES = False
TRAIN_MODEL = True

# =============================================
# FOLDER NAMES
ORIGINAL_DATA_FOLDER_NAME = "original_data"
MODELS_FOLDER_NAME = "models"
CHECKPOINT_FOLDER_NAME = "checkpoint"
CONFIG_FOLDER_NAME = "config"
INPUT_FOLDER_NAME = "input"
TOKENIZER_FOLDER_NAME = "tokenizer"
OUTPUT_DATA_FOLDER_NAME = "output_data"

HEMKIT_FOLDER_NAME = "HEMKit"
SOFTWARE_FOLDER_NAME = "software"
HIERARCHICAL_FILES_FOLDER_NAME = "hierarchical_files"
# =============================================


# =============================================
# FILE NAMES
LABEL_FILE_NAME = "labels.csv"

ORIGINAL_DATASET_NAME = "dataset.csv"
CLEANED_DATASET_NAME = "cleaned_data.csv"
BALANCED_DATASET_NAME = "balanced_data.csv"

TRAIN_FILE_NAME = "train.csv"
VALIDATION_FILE_NAME = "validation.csv"
TEST_FILE_NAME = "test.csv"
TEST_CLEANED_FILE_NAME = "test_cleaned.csv"

METRICS_FILE_NAME = "metrics.csv"

CONFIG_FILE_NAME = "config.json"
TRAIN_CONFIG_FILE_NAME = "train_config.json"

HEMKIT_FILE_NAME = "HEMKit.exe"
HIERARCHY_FILE_NAME = "hierarchy.txt"
GOLD_STANDARD_FILE_NAME = "gold_standard.txt"
PREDICTED_FILE_NAME = "predicted.txt"
# =============================================

# =============================================
# STRUCTURED GENERATION FIELD NAMES AND CONSTANTS
DECODER_ATTENTION_HEADS_FIELD = "decoder_attention_heads"
DECODER_ATTENTION_HEADS_VALUE = 8
DECODER_FFN_DIM_FIELD = "decoder_ffn_dim"
DECODER_FFN_DIM_VALUE = 2048
DECODER_DROPOUT_FIELD = "decoder_dropout"
DECODER_DROPOUT_VALUE = 0.1
RETURN_CLS_TOKEN_FIELD = "return_cls_token"
