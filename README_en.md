[View Chinese Version (查看中文版)](README.md)

# Mahjong Tile AI Clustering Tool

## Important Disclaimer

**This project serves solely as a technical demonstration and for educational purposes, explaining the process of training an AI model for Mahjong tile clustering. The project itself does not provide any pre-trained model weights (`.pth` files) or a ready-to-use Mahjong AI for practical application. Users need to collect their own data and complete the model training steps themselves. Furthermore, please note that the method described in this document is not the only way to achieve Mahjong tile clustering; various other techniques and strategies exist to accomplish similar goals.**

## Project Goal

This tool uses AI technology to automatically cluster captured Mahjong tile images using unsupervised learning. The goal is to start from scratch, collect data, train a model, and finally classify unknown Mahjong tile images (from the `capture/` directory) into different clusters within the `clustered_tiles/` directory. The project targets three-player Mahjong, aiming to divide the tiles into 29 categories.

## Project Structure

```
.
├── capture/              # Stores Mahjong image files **to be classified**
├── clustered_tiles/      # Stores **clustering results**, each subdirectory represents a cluster
├── configs/
│   ├── roi_config.json   # Region of Interest (ROI) config file
│   ├── tile_mapping.json # (Potential) Tile name/ID mapping file
│   └── tile_slicer_config.json # Tile slicing config file
├── data/                 # Stores manually classified Mahjong image files **for model training** (contains subdirs for each class)
├── gui/
│   ├── __init__.py       # GUI package identifier
│   ├── control_panel.py  # Control panel UI component
│   ├── image_display.py  # Image display UI component
│   └── main_window.py    # Main application window UI (for image collection & display)
├── image_processing/
│   ├── __init__.py       # Image processing package identifier
│   ├── hand_detector.py  # Hand detection related logic
│   ├── tile_defs.py      # Tile definitions or constants
│   └── tile_slicer.py    # Tile slicing logic
├── image_utils/
│   └── screen_capture.py # Screen capture utility functions
├── tools/
│   ├── cluster_captured_tiles.py # Script to execute image clustering (includes fallback method)
│   ├── find_roi_interactively.py # Tool to find ROI interactively
│   ├── interactive_tile_slicer.py # Tool for interactive tile slicing
│   └── restore_capture_from_clusters.py # (Potential) Tool to restore capture dir from clusters
├── trainer/
│   └── clustering/
│       # (May contain __init__.py, dataset.py, etc.)
│       ├── model.py      # CNN model definition
│       ├── train.py      # Script to train the CNN model
│       └── result/
│           └── best_mahjong_feature_extractor.pth # Trained best model weights
├── main.py               # Main application entry point, launches GUI to assist image collection etc.
├── README.md             # Chinese version of this documentation
└── README_en.md          # This documentation file (English)
# --- The following files are commonly present but not shown by list_dir; manual confirmation/addition is recommended ---
# ├── requirements.txt      # Lists required Python dependencies
# ├── .gitignore            # Specifies intentionally untracked files that Git should ignore
```

## Complete Workflow From Scratch

Below are the complete steps from having no data or model to obtaining the final clustering results:

### Step 1: Environment Setup

1.  **Python**: Ensure Python is installed (version 3.8 or higher recommended).
2.  **Dependencies**: Install the required Python libraries. Open a terminal or command prompt and run:
    ```bash
    pip install torch torchvision opencv-python numpy scikit-learn Pillow
    ```
    *   **Important**: For optimal performance (especially when using a GPU for training), refer to the [official PyTorch website](https://pytorch.org/) to install `torch` and `torchvision` according to your operating system, Python version, and CUDA version (if you have an NVIDIA GPU with CUDA installed).

**Purpose of this step:** To prepare the necessary foundation for the project to run. Python is the programming language itself, while the dependencies provide crucial functionalities: `torch` and `torchvision` for deep learning model training and image processing, `opencv-python` for image reading and basic operations, `numpy` for numerical computation, `scikit-learn` for clustering algorithms and evaluation, and `Pillow` for image handling.

### Step 2: Create Initial Training Data using Preliminary Clustering

The goal of this step is to create the first version of a labeled training dataset (`./data/`) starting from completely unlabeled images, reducing the burden of purely manual classification.

1.  **Collect Initial Unlabeled Images**: Place all the Mahjong tile images you have collected (without manual sorting) into the `capture/` directory first.
    *   **Tip:** You can run `python main.py` to launch the graphical user interface (GUI). This interface likely includes features like screen capture, hand region selection (ROI), and tile slicing, which can **greatly assist you** in collecting and extracting individual Mahjong tile images from game screens or other sources and saving them to the `capture/` directory.
2.  **Perform Preliminary Clustering (Using Fallback Method)**: Run the clustering script. Since there is no custom model yet, the script will automatically use the fallback method (ResNet18 + color features) for clustering.
    ```bash
    python tools/cluster_captured_tiles.py
    ```
3.  **Review and Organize Preliminary Results**: After the script finishes, preliminary clustering results (e.g., `cluster_00`, `cluster_01`, ...) will be generated in the `clustered_tiles/` directory. Now, **manual review and organization** are required:
    *   Create the `./data` directory (if it doesn't exist).
    *   Under `./data`, create corresponding subdirectories for each Mahjong tile type (29 types in total), e.g., `1m`, `Aka5p`, `N`, etc.
    *   **Examine each** `cluster_XX` subdirectory under `clustered_tiles/` one by one. Ideally, a cluster should mostly contain the same type of tile.
    *   **Move** the images **confirmed to belong to the same tile type** from each `cluster_XX` directory to the **correct category subdirectory** under `./data/`. For example, if `cluster_05` mostly contains "1 Man" tiles, move all "1 Man" images from it to `./data/1m/`.
    *   For obviously misclassified images (e.g., a "North Wind" tile mixed in the "1 Man" cluster), move them to the correct category subdirectory under `./data/` (e.g., `./data/N/`). You can discard images that are of too poor quality or unrecognizable for now.
4.  **Complete First Version of Training Set**: Once you have organized all usable images from `clustered_tiles/`, the `./data/` directory now contains your first version of the training dataset with (relatively) accurate labels.

**Purpose of this step:** To address the lack of labeled data when starting from scratch. By leveraging the machine's preliminary clustering ability, the heavy task of full manual classification is transformed into a task of **review and correction**, significantly improving the efficiency of creating the initial training set. The resulting `./data/` directory forms the basis for training the custom model later.

### Step 3: Train the First Custom Feature Extraction Model

Use the first version of the `./data/` dataset created in the previous step to train your first custom CNN model.

1.  **Run Training Script**: Ensure `--data_dir` points to your organized `./data` directory.
    ```bash
    python trainer/clustering/train.py --data_dir ./data --output_dir trainer/clustering --epochs 150 --batch_size 128 --lr 5e-4 --temperature 0.1 --image_size 96 --num_workers 4 --seed 42 --eval_freq 10 --save_freq 50
    ```
    *   (Parameter explanations as before)

2.  **Obtain the First Custom Model**: After training completes, the best model weights will be saved to `trainer/clustering/result/best_mahjong_feature_extractor.pth`.

**Purpose of this step:** To generate the first feature extractor optimized for Mahjong tiles based on the initially curated data. Although the training data might not be perfect yet, this model will typically perform better than the generic ResNet18, laying the groundwork for more accurate clustering later.

### Step 4: Prepare New Images for Classification

Collect **new** Mahjong tile images that you want to automatically classify using the **already trained custom model**.

1.  **Clear or Fill `capture/` Directory**: Place these **new, unlabeled** images into the `capture/` directory.

**Purpose of this step:** To provide the actual input data that needs to be classified using the **optimized model**.

### Step 5: Execute Image Clustering using the Custom Model

Utilize the custom model trained in Step 3 to classify the new images prepared in Step 4.

1.  **Confirm Model Existence**: Ensure the `trainer/clustering/result/best_mahjong_feature_extractor.pth` file exists.
2.  **Run Clustering Script**: Run the clustering script again.
    ```bash
    python tools/cluster_captured_tiles.py
    ```
3.  **Processing**: This time, the script will:
    *   **Successfully load** the custom model trained in Step 3.
    *   Read the **new** images from `capture/`.
    *   Extract feature vectors for each new image using the **custom model**.
    *   Cluster these feature vectors into 29 groups using `AgglomerativeClustering`.
    *   Copy the new images from `capture/` to the corresponding cluster subdirectories under `clustered_tiles/`.

**Purpose of this step:** To apply the **optimized custom model** to new, unseen data, expecting more accurate classification results than the preliminary clustering in Step 2.

**Fallback Clustering Method Explanation:**
*   This method is primarily used during the **initial data curation phase in Step 2**. While the script will still fall back to this method if the custom model fails to load, the expectation in the normal iterative workflow (Step 5) is to use the custom model.

### Step 6: Review Results and Iterate for Optimization

*   Check the latest clustering results in the `clustered_tiles/` directory, which were generated **using the custom model**.
*   Evaluate the clustering performance. These results should generally be more accurate than the preliminary ones from Step 2.
*   **(Optional) Iterative Optimization**: If you are satisfied with the results, you can consider **adding** the correctly classified images from this `clustered_tiles/` run back into the corresponding categories in the `./data/` directory. This **augments and refines** your training dataset. You can then optionally go back to **Step 3**, retrain the model using the updated `./data/`, potentially achieving further performance improvements. Repeating this cycle (Step 3 -> Step 4 -> Step 5 -> Step 6 -> Update Data -> Step 3...) allows for continuous model optimization.

**Purpose of this step:** To evaluate the custom model's performance and provide a pathway for continuous improvement. By repeatedly processing new data with the model and feeding the validated results back into the training set, the model's accuracy and robustness can be progressively enhanced.

## Notes

*   **Data Quality is Key**: Careful review and labeling of data, both during initial curation (Step 2) and subsequent iterations (Step 6), is crucial for improving model performance.
*   **Training Time**: Model training in Step 3 can take a significant amount of time.
*   **Iterative Improvement**: As described in Step 6, retraining with new data and the model is a common method to enhance results.