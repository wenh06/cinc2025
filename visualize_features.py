import argparse
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from umap import UMAP

from cfg import CFG
from const import SampleType
from dataset import CINC2025Dataset
from models import CRNN_CINC2025, FM_CINC2025

# Suppress Numba TBB warning
os.environ["NUMBA_WARNINGS"] = "0"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off warning for onednn from TensorFlow
print(f"Using device: {DEVICE}")


def get_args():
    parser = argparse.ArgumentParser(description="Visualize ECG Features using UMAP for FM and CRNN models.")

    parser.add_argument(
        "--model-arch",
        type=str,
        required=True,
        choices=["fm", "crnn"],
        dest="model_arch",
        help="Type of model to visualize: 'fm' (Foundation Model) or 'crnn' (CNN-RNN).",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, dest="checkpoint", help="Path to the trained model checkpoint."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        dest="dataset_dir",
        help="Path to the dataset. If not provided, will use default from config.",
    )
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size", help="Batch size for feature extraction.")
    parser.add_argument("--n-neighbors", type=int, default=15, dest="n_neighbors", help="UMAP n_neighbors parameter.")
    parser.add_argument("--min-dist", type=float, default=0.1, dest="min_dist", help="UMAP min_dist parameter.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        dest="save_path",
        help="Path to save the plot. Defaults to umap_{model_arch}.{format} in the images directory. format will be pdf and svg.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        dest="test_only",
        help="Run the script in test mode with a small subset of data for quick execution.",
    )

    return parser.parse_args()


def load_model(model_arch: Literal["fm", "crnn"], checkpoint_path: str) -> Tuple[torch.nn.Module, CFG]:
    """
    Load the specified model architecture and weights.

    Parameters
    ----------
    model_arch : {"fm", "crnn"}
        The model architecture to load: "fm" for Foundation Model, "crnn" for CRNN.
    checkpoint_path : str
        Path to the .pth file.

    Returns
    -------
    Tuple[torch.nn.Module, CFG]
        The loaded model in eval mode on DEVICE,
        and the training configuration.

    """
    print(f"Initializing {model_arch.upper()} model...")
    print(f"Loading weights from {checkpoint_path}...")

    if model_arch == "fm":
        model, train_config = FM_CINC2025.from_checkpoint(checkpoint_path, weights_only=False)
        train_config.fs = model.config.fs[model.config.name]
        train_config.resample.fs = model.config.fs[model.config.name]
    elif model_arch == "crnn":
        model, train_config = CRNN_CINC2025.from_checkpoint(checkpoint_path, weights_only=False)
        train_config.fs = model.config.fs
        train_config.resample.fs = model.config.fs
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    model.to(DEVICE)
    model.eval()

    return model, CFG(train_config)


def extract_features(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract features using a forward hook on the classification layer input.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    dataloader : DataLoader
        The data loader.

    Returns
    -------
    features : np.ndarray
        Extracted features matrix (N_samples, Embed_Dim).
    labels : np.ndarray
        Target labels (N_samples,).
    types : np.ndarray or None
        Sample types/sources (N_samples,).

    """
    features_list = []
    labels_list = []
    types_list = []

    # Identify Hook Point
    # Use isinstance for robust type checking instead of hasattr
    if isinstance(model, FM_CINC2025):
        target_layer = model.head
        print("Hooking into 'model.head' (FM architecture)")
    elif isinstance(model, CRNN_CINC2025):
        target_layer = model.clf
        print("Hooking into 'model.clf' (CRNN architecture)")
    else:
        # Fallback: try to print modules to debug
        raise AttributeError("Unknown model type. Could not find classification head ('head' or 'clf') in model.")

    # Register Hook
    hook_data = {}

    def hook_fn(module, input, output):
        # input is a tuple of args passed to the layer.
        # The first arg is the feature tensor [Batch, Dim]
        hook_data["feat"] = input[0].detach().cpu()

    handle = target_layer.register_forward_hook(hook_fn)
    print("Registered forward hook for feature extraction.")
    print(f"Extracting features on device: {model.device}")
    # Inference Loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features", dynamic_ncols=True):
            signals = batch["signals"]
            labels = batch["chagas"]
            demographics = batch.get("demographics", None)
            sample_types = batch.get("sample_type", None)

            # Prepare inputs
            input_tensors = {"signals": signals.to(DEVICE)}

            # Demographic handling
            if hasattr(model.config, "dem_encoder") and model.config.dem_encoder.enable:
                assert demographics is not None, "Model expects demographic data but it's missing in the batch."
                input_tensors["demographics"] = demographics.to(DEVICE)

            # Forward (Hook captures features)
            _ = model(input_tensors)

            # Store data
            if "feat" in hook_data:
                features_list.append(hook_data["feat"].numpy())
            else:
                raise RuntimeError(
                    "Feature extraction hook was not triggered. Ensure the correct layer is hooked and executed in forward pass."
                )

            # Handle labels (One-hot -> Scalar)
            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=-1)
            labels_list.append(labels.cpu().numpy())

            if sample_types is not None:
                types_list.append(sample_types.cpu().numpy())

    handle.remove()

    # Concatenate
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    if types_list:
        types = np.concatenate(types_list, axis=0)
    else:
        types = None

    return features, labels, types


def plot_umap(embedding, labels, types, title, save_path):
    """Plot UMAP with custom palette for SampleTypes."""

    if fm.findfont("Times New Roman", fallback_to_default=False):
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        print("Times New Roman font not found. Using default font.")

    df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])

    # Mapping logic (Same as before)
    if types is not None:
        type_map = {
            SampleType.NEGATIVE_SAMPLE.value: "Negative (PTB-XL)",
            SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value: "Positive (Self-Reported/Code-15%)",
            SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value: "Uncertain (Self-Reported/Code-15%)",
            SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value: "Positive (Serology-Confirmed/SaMi-Trop)",
        }
        df["Source"] = [type_map.get(int(t), str(t)) for t in types]
        hue_col = "Source"

        # Color Palette
        palette = {
            "Negative (PTB-XL)": "#27ae60",  # Green
            "Uncertain (Self-Reported/Code-15%)": "#2980b9",  # Blue
            "Positive (Self-Reported/Code-15%)": "#f39c12",  # Orange
            "Positive (Serology-Confirmed/SaMi-Trop)": "#c0392b",  # Red
        }
        # Filter palette
        unique_sources = df["Source"].unique()
        palette = {k: v for k, v in palette.items() if k in unique_sources}
    else:
        df["Label"] = ["Positive" if lb == 1 else "Negative" for lb in labels]
        hue_col = "Label"
        palette = {"Negative": "grey", "Positive": "red"}

    plt.figure(figsize=(10, 12))

    # Determine sizes (highlight confirmed cases)
    # Use a column for size mapping
    df["Size"] = [40 if "Confirmed" in str(s) else 15 for s in df[hue_col]]

    sns.scatterplot(
        data=df, x="UMAP1", y="UMAP2", hue=hue_col, style=hue_col, palette=palette, size="Size", sizes=(15, 40), alpha=0.7
    )

    # plt.title(title, fontsize=20)
    plt.xlabel("UMAP1", fontsize=16)
    plt.ylabel("UMAP2", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2, fontsize=14, title_fontsize=16)
    plt.tight_layout()

    if save_path is None:
        save_path_pdf = f"images/umap_{title.split(':')[1].strip().split()[0].lower()}.pdf"
        plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path_pdf}")

        save_path_svg = f"images/umap_{title.split(':')[1].strip().split()[0].lower()}.svg"
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path_svg}")
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


def plot_score_distribution(model, dataloader, ylim=None, save_path=None):
    """
    Plot KDE distribution of predicted scores, matching the style and palette of the UMAP plot.
    Handles 4 sample types.
    """

    if fm.findfont("Times New Roman", fallback_to_default=False):
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        print("Times New Roman font not found. Using default font.")

    if isinstance(model, FM_CINC2025):
        model_arch = "fm"
    elif isinstance(model, CRNN_CINC2025):
        model_arch = "crnn"
    else:
        raise ValueError("Unknown model type for score distribution plotting.")

    model.eval()
    scores = []
    types = []
    device = model.device

    print("Extracting scores for distribution plot...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Scores", dynamic_ncols=True):
            sigs = batch["signals"].to(device)
            batch_types = batch["sample_type"]
            demographics = batch.get("demographics", None)
            if demographics is not None:
                demographics = demographics.to(device)

            out = model({"signals": sigs, "demographics": demographics})
            probs = out["chagas_prob"][:, 1]

            scores.extend(probs.cpu().numpy())
            types.extend(batch_types.numpy())

    df = pd.DataFrame({"Score": scores, "Type Code": types})
    df["Type Code"] = df["Type Code"].astype(int)

    type_map = {
        SampleType.NEGATIVE_SAMPLE.value: "Negative (PTB-XL)",
        SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value: "Positive (Self-Reported/Code-15%)",
        SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value: "Uncertain (Self-Reported/Code-15%)",
        SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value: "Positive (Serology-Confirmed/SaMi-Trop)",
    }

    df["Source"] = df["Type Code"].map(type_map)
    df = df.dropna(subset=["Source"])
    if len(df) == 0:
        raise ValueError("No valid data after cleaning! Check Type Code mapping.")

    palette = {
        "Negative (PTB-XL)": "#27ae60",  # Green
        "Uncertain (Self-Reported/Code-15%)": "#2980b9",  # Blue
        "Positive (Self-Reported/Code-15%)": "#f39c12",  # Orange
        "Positive (Serology-Confirmed/SaMi-Trop)": "#c0392b",  # Red
    }

    valid_sources = df["Source"].unique()
    current_palette = {k: v for k, v in palette.items() if k in valid_sources}
    print(f"Valid color palette: {current_palette}")

    plt.figure(figsize=(10, 6))

    ax = sns.kdeplot(
        data=df,
        x="Score",
        hue="Source",
        palette=current_palette,
        fill=True,
        common_norm=False,
        alpha=0.3,
        linewidth=2.5,
        legend=False,
    )
    desired_order = [
        "Negative (PTB-XL)",
        "Uncertain (Self-Reported/Code-15%)",
        "Positive (Self-Reported/Code-15%)",
        "Positive (Serology-Confirmed/SaMi-Trop)",
    ]
    ordered_sources = [s for s in desired_order if s in valid_sources]

    handles = []
    labels = []
    for source in ordered_sources:
        handle = plt.Line2D([], [], color=current_palette[source], linewidth=2.5, alpha=0.8, label=source)
        handle_fill = plt.Rectangle((0, 0), 1, 1, facecolor=current_palette[source], alpha=0.3)
        handles.append((handle, handle_fill))
        labels.append(source)

    ax.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        title="Data Source",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        title_fontsize=14,
        frameon=True,
    )

    plt.title("Distribution of Predicted Risk Scores by Data Source", fontsize=18)
    plt.xlabel("Predicted Chagas Probability", fontsize=14)
    plt.ylabel("Density (Normalized per Group)", fontsize=14)
    plt.xlim(-0.01, 1)
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is None:
        Path("images").mkdir(exist_ok=True)
        save_path_pdf = f"images/score_distribution_{model_arch}.pdf"
        plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path_pdf}")

        save_path_svg = f"images/score_distribution_{model_arch}.svg"
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path_svg}")
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def run_visualization(
    model_arch: Literal["fm", "crnn"],
    checkpoint: str,
    dataset_dir: str,
    batch_size: int = 128,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save_path: Optional[str] = None,
    test_only: bool = False,
):
    """
    Run the UMAP visualization pipeline.

    Parameters
    ----------
    model_arch : str
        "fm" or "crnn".
    checkpoint : str
        Path to model checkpoint.
    dataset_dir : str
        Path to dataset directory.
    batch_size : int, optional
        Inference batch size.
    n_neighbors : int, optional
        UMAP parameter.
    min_dist : float, optional
        UMAP parameter.
    save_path : str, optional
        Path to save the plot. If None, saves as pdf/svg in images/.
    test_only : bool, default False
        If True, runs in test mode with a smaller subset of data for quick execution.

    """

    # Load Model
    model, train_config = load_model(model_arch, checkpoint)

    # Dataset & Sampling (Balance is key for visualization)
    print("Loading Dataset...")
    if test_only:
        train_config.extra_experiment = True
        dataset = CINC2025Dataset(train_config, training=False, db_dir=dataset_dir, part="test")
        # Filter DF to only include test records
        df = dataset.reader._df_records[dataset.reader._df_records.index.isin(dataset.records)].copy()
    else:
        dataset = CINC2025Dataset(train_config, training=False, db_dir=dataset_dir)
        df = dataset.reader._df_records

    # Identify indices by type for sampling
    idx_map = {sample_type.value: df.index[df["sample_type"] == sample_type.value].tolist() for sample_type in SampleType}

    # Sampling strategy: All Positives + Sampled Negatives
    # Keep visualization clean, restrict negative class sample size
    sel_indices = []
    sel_indices.extend(idx_map[SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value])  # All Confirmed Pos
    sel_indices.extend(idx_map[SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value])  # All Self-Reported Pos
    n_pos = len(sel_indices)
    n_neg = int(n_pos * 1.2)  # Aim for ~1.2x negatives to positives for better visualization balance

    if len(idx_map[SampleType.NEGATIVE_SAMPLE.value]) > 0:
        sel_indices.extend(
            np.random.choice(
                idx_map[SampleType.NEGATIVE_SAMPLE.value],
                min(len(idx_map[SampleType.NEGATIVE_SAMPLE.value]), n_neg // 2),
                replace=False,
            )
        )
    if len(idx_map[SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value]) > 0:
        sel_indices.extend(
            np.random.choice(
                idx_map[SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value],
                min(len(idx_map[SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value]), n_neg // 2),
                replace=False,
            )
        )
    n_neg = len(sel_indices) - n_pos

    print(f"Selected {len(sel_indices)} samples ({n_pos} positives, {n_neg} negatives) for visualization.")

    dataset._reset_records(sel_indices)  # Reset dataset to only include selected indices
    if isinstance(model, FM_CINC2025):
        print("Using FM_CINC2025 model, adjusting input_len (if necessary for ST-MEM backbone)...")
        dataset.reset_input_len(model.config.input_len[model.config.name], reload=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Extract
    features, labels, types_batch = extract_features(model, dataloader)

    # Fallback if types missing in batch (retrieve from DF via indices)
    if types_batch is None:
        print("Retrieving sample types from DataFrame...")
        types = df.loc[sel_indices, "sample_type"].values
    else:
        types = types_batch

    # UMAP
    print("Running UMAP...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine")
    embedding = reducer.fit_transform(features)

    # Plot
    plot_umap(embedding, labels, types, title=f"UMAP Features: {model_arch.upper()} Model", save_path=save_path)


def main():
    args = get_args()
    run_visualization(
        model_arch=args.model_arch,
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        save_path=args.save_path,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
