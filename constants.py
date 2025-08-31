
# --- Core definitions for DINO v2/v3 classifiers ------------

image_extensions = ["png", "jpg", "jpeg", "tif", "tiff"]

trn_folder = "TRN"
val_folder = "VAL"
tst_folder = "TST"
partition_folders = [trn_folder, val_folder, tst_folder]

FORMAT_TYPES = ["dv2_best", "dv2_final"]

RESULTS_FILE_NAME = "learn_curves.csv"
TEST_RESULTS_FILE_NAME = "test_eval_results.txt"
