[查看中文版 (View Chinese Version)](README.md)

# Mahjong Tile AI Image Analysis Tool (Clustering & Classification)

## Important Disclaimer

**This project serves solely as a technical demonstration and educational resource, explaining the process of training AI models for Mahjong tile image analysis (including clustering and classification). The project itself does not provide any pre-trained model weights (`.pth` files) or a ready-to-use Mahjong AI application. Users need to collect their own data and complete the model training steps. Furthermore, please note that the methods described in this document are not the only way to achieve Mahjong tile image analysis; various different techniques and strategies can accomplish similar goals.**

## Project Goal

This tool aims to demonstrate using AI technology for automatic analysis of captured Mahjong tile images, including using unsupervised **clustering** to assist data organization and training **classification** models to identify tiles. The goal is to start from scratch, collect data, train models, and ultimately classify unknown Mahjong tile images (from the `capture/` directory) into different clusters and use trained classifiers for precise identification and organization. The project targets three-player Mahjong, aiming to classify tiles into 29 categories.

## Project Structure

```
.
├── capture/              # Stores Mahjong tile image files **to be classified/clustered**
├── clustered_tiles/      # Stores **clustering results**, each subdirectory represents a cluster
├── configs/
│   ├── roi_config.json   # Region of Interest (ROI) configuration file
│   ├── tile_mapping.json # (Possible) Tile name/ID mapping file
│   └── tile_slicer_config.json # Tile slicing configuration file
├── data/                 # Stores manually classified tile image files **for model training** (contains subdirectories for each class, including noise)
├── gui/
│   ├── __init__.py       # GUI package identifier
│   ├── control_panel.py  # Control panel UI component
│   ├── image_display.py  # Image display UI component
│   └── main_window.py    # Main application window UI (for image collection and display)
├── image_processing/
│   ├── __init__.py       # Image processing package identifier
│   ├── hand_detector.py  # Hand detection related logic
│   ├── tile_defs.py      # Tile definitions or constants
│   └── tile_slicer.py    # Tile slicing logic
├── image_utils/
│   └── screen_capture.py # Screen capture utility function
├── test_inference/       # Directory for storing **classification model inference results**
├── tools/
│   ├── cluster_captured_tiles.py # Script to execute image clustering (includes fallback method)
│   ├── find_roi_interactively.py # Tool for interactively finding the ROI
│   ├── infer_and_organize.py     # Script to infer and organize files using the classification model
│   └── interactive_tile_slicer.py # Interactive tile slicing tool
├── trainer/
│   ├── clustering/       # Clustering model training related
│   │   ├── losses.py     # Clustering loss function definitions (e.g., SupConLoss)
│   │   ├── model.py      # CNN model definition for clustering (e.g., SupConResNet)
│   │   ├── train.py      # Script to train the clustering feature extraction model
│   │   └── result/       # Stores clustering model training results
│   │       └── ... .pth    # Trained clustering model weights
│   └── classification/   # Classification model training related
│       ├── dataset.py    # Classification dataset handling
│       ├── eval.py       # Script to evaluate the classification model
│       ├── model.py      # CNN model definition for classification (e.g., SimpleMahjongCNN)
│       ├── train.py      # Script to train the classification model
│       └── results/      # Stores classification model training results
│           ├── logs/       # TensorBoard logs
│           └── models/     # Trained classification model weights
│               └── ... .pth
├── .gitignore            # Specifies files/directories Git should ignore
├── LICENSE               # Project license file
├── main.py               # Main application entry point, launches the GUI for assisted image collection etc.
├── README.md             # This documentation file (Chinese)
└── README_en.md          # English version documentation file
└── requirements.txt      # Lists required Python dependencies for the project
```

## Complete Workflow from Scratch

Below are the complete steps from having no data and models to finally being able to train and use clustering and classification models:

### Step 1: Environment Setup

1.  **Python**: Ensure Python is installed (version 3.8 or higher recommended).
2.  **Dependencies**: Install the required Python libraries. Open a terminal or command prompt and run:
    ```bash
    pip install torch torchvision opencv-python numpy scikit-learn Pillow tensorboard
    ```
    *   **Important**: For optimal performance (especially leveraging GPU during training), refer to the instructions on the [PyTorch official website](https://pytorch.org/) to install `torch` and `torchvision` based on your operating system, Python version, and CUDA version (if you have an NVIDIA GPU with CUDA installed).

**Purpose of this step:** Prepare the necessary runtime foundation for the project. Python is the programming language itself, while the dependencies provide key functionalities: `torch` and `torchvision` for deep learning model training and image processing, `opencv-python` for image reading and basic operations, `numpy` for numerical computations, `scikit-learn` for clustering algorithms and evaluation, `Pillow` for image handling, and `tensorboard` for visualizing the training process.

### Step 2: Create Initial Training Data Base using Preliminary Clustering

The goal of this step is to start with completely unlabeled images and use a **pre-defined clustering method** to create a preliminarily classified dataset, reducing the burden of subsequent manual labeling and organization.

1.  **Collect Initial Unclassified Images**: Place all collected Mahjong tile images (no need for manual classification yet) into the `capture/` directory.
    *   **Tip:** You can run `python main.py` to launch the graphical user interface (GUI). This interface might include features like screen capture, hand region selection (ROI), and tile slicing, which can **greatly assist** you in collecting and slicing individual Mahjong tile images from game screens or other sources and saving them to the `capture/` directory.
2.  **Perform Preliminary Clustering (Using Fallback Method)**: Run the clustering script. Since there is no custom model yet, the script will **automatically use the fallback method** (based on pre-trained ResNet18 + color features) for clustering.
    ```bash
    # Ensure running from the project root directory
    python tools/cluster_captured_tiles.py
    ```
3.  **Review and Organize Preliminary Results**: After the script finishes, the `clustered_tiles/` directory will contain the initial clustering results (e.g., `cluster_00`, `cluster_01`, ...). **Manual review and organization** are now required to prepare the base data for training the **custom clusterer** and **classifier** later:
    *   Create the `./data` directory (if it doesn't exist).
    *   Inside `./data`, create subdirectories for each type of Mahjong tile (target is 29 types currently, e.g., `1m`, `Aka5p`, `N`, etc.). Also, create a `noise` subdirectory (`./data/noise/`) for non-Mahjong tile images (e.g., background, special effects glow, unrecognizable images, etc.).
    *   **Examine** each `cluster_XX` subdirectory under `clustered_tiles/` one by one. Ideally, a cluster should mostly contain the same type of tile.
    *   **Move** the images **confirmed to be of the same tile type** from each `cluster_XX` directory to the corresponding **correct class subdirectory** under `./data/`. For example, if `cluster_05` mostly contains "1 Man" tiles, move all "1 Man" images from it to `./data/1m/`.
    *   For obviously misclassified images (e.g., a "North Wind" tile mixed in the "1 Man" cluster), move it to the correct class subdirectory under `./data/` (e.g., `./data/N/`).
    *   If you find non-Mahjong tile images (like backgrounds, flashes) incorrectly clustered, move them to the `./data/noise/` directory. Poor quality or unrecognizable images can also be placed in `noise` or discarded.
4.  **Complete First Version of Training Set**: Once you have organized all usable images from `clustered_tiles/`, the `./data/` directory contains your first version of the training dataset with (relatively) accurate labels.

**Purpose of this step:** Address the lack of labeled data when starting from scratch. By leveraging the machine's preliminary clustering capability, the heavy task of full manual classification is transformed into a task of **reviewing and correcting**, significantly improving the efficiency of creating the initial training set. The resulting `./data/` directory is the foundation for training custom clustering and classification models later.

### Step 3: Train the First Custom Clustering Feature Extraction Model

Using the first version of the `./data/` dataset created in the previous step, train your first **custom clustering feature extraction model** (using Supervised Contrastive Learning). The goal of this model is to learn a good feature space where Mahjong tiles of the same class are closer together.

1.  **Run Clustering Training Script**: Ensure `--data_dir` points to your organized `./data` directory.
    *   **Important**: Due to internal package imports, make sure to run from the **project root directory** using `python -m`.
    ```bash
    # Must run from the project root directory
    python -m trainer.clustering.train --data_dir ./data --output_dir trainer/clustering --epochs 100 --batch_size 128 --lr 5e-4 --save_top_k 3 --num_workers 4 [other_parameters...]
    ```
    *   `--data_dir ./data`: Specifies the directory containing the classified training images.
    *   `--output_dir trainer/clustering`: Specifies the directory to save training logs and model files. Results will be saved in the `result/` subdirectory within this directory (e.g., `trainer/clustering/result/`).
    *   `--epochs 100`: Number of training epochs (SupCon might need more).
    *   `--batch_size 128`: Batch size (contrastive learning often benefits from larger batches).
    *   `--lr 5e-4`: Learning rate.
    *   `--save_top_k 3`: Saves the top 3 models based on the validation set ARI metric (adjustable).
    *   `--num_workers 4`: Number of data loading workers. **Hint**: This value isn't always "the more, the better." It's recommended to experiment with different values (e.g., 0, 2, 4, 8...) based on your CPU cores and memory size to find the optimal training speed. Too high a value can sometimes slow down training due to process management overhead.
    *   *(Check the `trainer/clustering/train.py` script for more available parameters)*
2.  **Obtain Custom Clustering Model**: After training completes, the top K performing models will be saved in the `result/` subdirectory under the directory specified by `--output_dir`, with filenames like `model_epoch_XX_ari_Y.YYYY.pth`.

**Purpose of this step:** Generate the first **feature extractor** optimized for Mahjong tiles based on the preliminarily organized data. Although the training data might not be perfect yet, the features learned by this model are generally more suitable for subsequent Mahjong tile clustering tasks than the generic ResNet18.

### Step 4: Prepare New Images for Clustering

Collect the **new** Mahjong tile images that you want to automatically cluster using the **trained custom clustering model**.

1.  **Empty or Fill `capture/` Directory**: Place these **new, unclassified** images into the `capture/` directory.

**Purpose of this step:** Provide the actual input data that needs to be clustered using the **optimized model**.

### Step 5: Perform Image Clustering using the Custom Model

Utilize the **best custom clustering model** trained in Step 3 to cluster the new images prepared in Step 4.

1.  **Run Clustering Script**: Run the clustering script again, this time using the `--model_path` parameter to specify the path to the model file you want to use.
    ```bash
    # Replace <path_to_your_best_cluster_model.pth> with the actual model file path
    # e.g., trainer/clustering/result/model_epoch_XX_ari_Y.YYYY.pth
    python tools/cluster_captured_tiles.py --model_path <path_to_your_best_cluster_model.pth>
    ```
    *   **Note**: If the `--model_path` parameter is omitted, or if the provided path is invalid (file doesn't exist or cannot be loaded), the script will **automatically fall back** to using the alternative method (pre-trained ResNet18 + color features) for clustering and display a warning in the logs.
2.  **Processing**: If a valid model path is provided, the script will now:
    *   **Successfully load** the custom clustering model specified via `--model_path`.
    *   Read the **new** images from `capture/`.
    *   Extract feature vectors for each new image using the **custom model**.
    *   Cluster these feature vectors into **30** clusters using `AgglomerativeClustering` (because we include noise).
    *   Copy the new images from `capture/` to the corresponding cluster subdirectories under `clustered_tiles/` based on the clustering results.

**Purpose of this step:** Apply the **optimized custom clustering model** to process new, unknown data, expecting more accurate clustering results than the preliminary clustering in Step 2.

### Step 6: Review Results and Iteratively Optimize the Clusterer

*   Check the latest clustering results obtained **using the custom clustering model** in the `clustered_tiles/` directory.
*   Evaluate the clustering performance. Usually, these results will be more accurate than the preliminary results from Step 2.
*   **(Optional) Iteratively Optimize the Clusterer**: If you want to further improve the **clustering model's** effectiveness, carefully review the results in `clustered_tiles/` this time. **Add** the **correctly** classified images (including tiles correctly clustered into a tile group and noise images correctly clustered into the noise group) back to the `./data/` directory according to their **true class** (including the `noise` class) to **expand and refine** your training dataset. Then, you can optionally return to **Step 3**, use the updated `./data/` to **retrain the clustering model**, aiming for further performance improvement. Repeating this process (Step 3 -> Step 4 -> Step 5 -> Step 6 -> Update Data -> Step 3...) can continuously optimize the clustering model.

**Purpose of this step:** Evaluate the effectiveness of the custom clustering model and provide a path for continuous improvement. By repeatedly processing new data with the model and feeding the verified results back into the training set, the feature extraction capability of the clustering model can be gradually enhanced.

### Step 7: Train Classification Model

After ensuring that the `./data/` directory contains training data with accurate labels (each subdirectory represents a class, **including a `noise` class**) through clustering-assisted organization and manual labeling, you can start training a **classification model**. The goal of this model is to directly predict which specific class an input image belongs to (e.g., "1m", "Wh", "noise").

1.  **Confirm Training Data**: Double-check the `./data/` directory and its subdirectories to ensure the data is organized by class and the `noise` directory contains representative non-Mahjong tile images.
2.  **Run Classification Training Script**: Execute the `trainer/classification/train.py` script.
    ```bash
    # Recommended to run from the project root directory
    python -m trainer.classification.train --data_dir ./data --output_dir trainer/classification/results --epochs 50 --batch_size 128 --lr 0.001 --save_top_k 3 --num_workers 4
    ```
    *   `--data_dir ./data`: Specifies the directory containing the classified training images.
    *   `--output_dir trainer/classification/results`: Specifies the directory to save training logs and model files. Results will be saved in the `models/` and `logs/` subdirectories within this directory.
    *   `--epochs 50`: Number of training epochs (adjustable).
    *   `--batch_size 128`: Batch size (adjust based on GPU memory).
    *   `--lr 0.001`: Learning rate.
    *   `--save_top_k 3`: Saves the top 3 models with the highest validation accuracy (adjustable).
    *   `--num_workers 4`: Number of data loading workers. **Hint**: Similar to clustering training, experimental adjustment based on hardware is recommended.
    *   *(Check the `trainer/classification/train.py` script for more available parameters)*
3.  **Obtain Classification Model**: After training completes, the top K performing models will be saved in the `models/` subdirectory under the directory specified by `--output_dir`, with filenames like `model_epoch_XX_acc_Y.YYYY.pth`.

**Purpose of this step:** Train a supervised learning model capable of making explicit class predictions for input Mahjong tile images (or non-Mahjong images). This model is the basis for subsequent automatic classification, inference, and organization tasks.

### Step 8: Utilize the Classifier for Subsequent Work (Inference and Organization)

Once the **classification model** is trained, you can use it to predict labels for a batch of new, unlabeled images and automatically organize these images into folders named after the predicted classes.

1.  **Prepare Images for Classification**: Place the images you want to classify into a folder, for example, the `./capture` directory.
2.  **Find the Trained Classification Model**: Locate the classification model file (`.pth` file) you want to use in the `trainer/classification/results/models/` directory, e.g., `model_epoch_XX_acc_Y.YYYY.pth`.
3.  **Run Inference and Organization Script**: Execute the `tools/infer_and_organize.py` script.
    ```bash
    python tools/infer_and_organize.py --model_path <path_to_your_model.pth> --input_dir ./capture --output_dir ./test_inference
    ```
    *   Replace `<path_to_your_model.pth>` with the actual path to your chosen classification model file.
    *   `--input_dir`: Specifies the directory containing the images to be classified (defaults to `./capture`).
    *   `--output_dir`: Specifies the directory to save the organized results (defaults to `./test_inference`, will be cleared if it already exists).
4.  **View Inference Results**: After the script finishes, check the `./test_inference/` directory. The script will create subdirectories for each predicted class (e.g., `1m`, `Wh`, `noise`, etc.) based on the model's predictions and copy the corresponding original images into these subdirectories.

**Purpose of this step:** Provide a method to directly use the **trained classification model** for predicting labels on new images and automatically organizing the files, facilitating quick review of the model's actual classification performance or for use in other subsequent applications.

## Notes

*   **Data Quality is Key**: Whether during initial organization (Step 2) or subsequent iterations (Step 6), carefully reviewing and labeling data is crucial for improving model performance.
*   **Training Time**: Model training (clustering feature extraction or classification) can take a significant amount of time, depending on the data volume and hardware.
*   **Iterative Improvement**: As mentioned in Step 6, iteratively training the **clustering model** with new data and model feedback is a common method to enhance its feature extraction effectiveness. Classification model training is typically done once the data labels are relatively stable.
*   **Distinguish Clustering and Classification**: Be aware of the different goals and training methods for Clustering (learning good feature representations to distinguish groups) and Classification (mapping input to known labels).