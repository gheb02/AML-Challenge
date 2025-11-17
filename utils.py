import argparse
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
from model import *

# roberta-large-nli-stsb-mean-tokens
def load_text_model(model_name="sentence-transformers/roberta-large-nli-stsb-mean-tokens"):
    """Load Sentence-BERT text encoder."""
    print(f"Loading text model: {model_name}")
    return SentenceTransformer(model_name)


def load_image_model(model_name="facebook/dinov2-giant"):
    """Load DINOv2 image encoder."""
    print(f"Loading image model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    return image_processor, model


@torch.inference_mode()
def process_images_batch(image_processor, model, image_paths, device, batch_size=128, dataset_path=None):
    """Generate image embeddings in batches."""
    print(f"Processing {len(image_paths)} images in batches...")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    failed_indices = []
    img_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        valid_images = []
        
        # Keep track of which original indices correspond to valid images in the batch
        valid_paths = []

        for j, path in enumerate(batch_paths):
            original_index = i + j
            try:
                img = Image.open(dataset_path / 'Images' / path).convert("RGB")
                valid_images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Skipping image {path} due to error: {e}")
                failed_indices.append(original_index)

        if not valid_images:
            continue

        inputs = image_processor(images=valid_images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # Average over patch tokens
        image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Store embeddings based on their success
        all_embeddings.extend(image_features)
        img_files.extend(valid_paths)
    
    if not all_embeddings:
        return np.array([]), list(range(len(image_paths)))

    return img_files, np.vstack(all_embeddings)

def process_captions(text_model, captions, device):
    """Generate text embeddings using Sentence-BERT."""
    print("Processing captions...")
    return text_model.encode(
        captions, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        device=device
    )

def load_dataset(dataset_path):
    """
    Load dataset from a directory containing captions.txt and an Images folder.
    """

    dataset_path = Path(dataset_path)
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "Images"

    if not captions_file.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Could not find 'captions.txt' or 'Images' directory in {dataset_path}")

    df = pd.read_csv(captions_file)
    
    if 'id' not in df.columns:
        df['id'] = np.arange(len(df))

    return df

def create_data_file(dataset_path, output_file, device=None, args={}):
    """
    Main function to generate embeddings and save the final .npz file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    text_model = load_text_model()
    image_processor, image_model = load_image_model()

    print(f"Loading dataset from: {dataset_path}")
    df_captions = load_dataset(dataset_path)
    all_captions = df_captions['caption'].tolist()
    caption2img = df_captions['image'].tolist()
    all_images = df_captions['image'].unique().tolist()

    num_images = len(all_images)
    num_captions = len(all_captions)
    print(f"Found {num_images} images and {num_captions} total captions.")

    all_images, img_embd = process_images_batch(image_processor, image_model, all_images, device, dataset_path=dataset_path)
    images_dict = {img_name: i for i, img_name in enumerate(all_images)}

    caption_embeddings = process_captions(text_model, df_captions['caption'].tolist(), device)
    
    label = np.zeros((num_captions, num_images), dtype=np.bool)
    for idx in range(num_captions):
        img_name = caption2img[idx]
        img_idx = images_dict[img_name]
        label[idx, img_idx] = 1

    data = {
        'metadata/num_captions': np.array([num_captions]),
        'metadata/num_images': np.array([num_images]),
        'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
        'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
        'captions/ids': df_captions['id'].to_numpy(),
        'captions/text': np.array(all_captions),
        'captions/embeddings': caption_embeddings,
        'captions/label': label,
        'images/names': np.array(all_images),
        'images/embeddings': img_embd,
    }

    print(f"Saving processed data to {output_file}")
    np.savez_compressed(output_file, **data)
    print("✓ Done.")
    
    if args.create_secret_version:
        data_secret = {
            'metadata/num_captions': np.array([num_captions]),
            'metadata/embedding_dim_text': np.array([caption_embeddings.shape[1]]),
            'metadata/embedding_dim_image': np.array([img_embd.shape[1]]),
            'captions/ids': df_captions['id'].to_numpy(),
            'captions/text': np.array(all_captions),
            'captions/embeddings': caption_embeddings,
        }

        secret_output_file = str(Path(output_file).with_suffix('.clean.npz'))
        print(f"Saving secret version to {secret_output_file}")
        np.savez_compressed(secret_output_file, **data_secret)
        print("✓ Secret version saved.")


def load_data(path):
    """Load processed data from .npz file"""
    data = dict(np.load(path, allow_pickle=True))
    # data['caption2img'] = data['caption2img'].item()
    # data['caption2img_idx'] = data['caption2img_idx'].item()
    return data



def prepare_train_data(data):
    """Prepare training data from loaded dict"""
    caption_embd = data['captions/embeddings']
    image_embd = data['images/embeddings']
    # Map caption embeddings to corresponding image embeddings
    label = data['captions/label'] # N x M

    # repeat the image embeddings according to the label
    label_idx = np.nonzero(label)[1]
    print(label_idx.shape)
    image_embd = image_embd[label_idx]
    assert caption_embd.shape[0] == image_embd.shape[0], "Mismatch in number of caption and image embeddings"

    X = torch.from_numpy(caption_embd).float()
    # Map each caption to its corresponding image embedding
    y = torch.from_numpy(image_embd).float()
    label = torch.from_numpy(label).bool()

    print(f"Train data: {len(X)} captions, {len(image_embd)} images")
    return X, y, label


def prepare_train_data_avg(data):
    """Prepara i dati di training mediando le caption per ogni immagine."""
    caption_embd = data['captions/embeddings']   # (N_captions, D)
    image_embd = data['images/embeddings']       # (N_images, D)
    label = data['captions/label']               # (N_captions, N_images)

    num_images = image_embd.shape[0]
    embedding_dim = caption_embd.shape[1]

    # Array per salvare la media delle caption per ogni immagine
    mean_caption_embds = np.zeros((num_images, embedding_dim))
    counts = np.zeros(num_images, dtype=int)

    # Per ogni caption, somma il suo embedding alla sua immagine
    for i in range(label.shape[0]):
        img_idx = np.nonzero(label[i])[0][0]  # trova immagine associata
        mean_caption_embds[img_idx] += caption_embd[i]
        counts[img_idx] += 1

    # Calcola la media vera e propria
    for j in range(num_images):
        if counts[j] > 0:
            mean_caption_embds[j] /= counts[j]

    # Ora abbiamo una media per immagine → mappiamo 1:1 con image_embd
    X = torch.from_numpy(mean_caption_embds).float()
    y = torch.from_numpy(image_embd).float()

    print(f"Train data (mean): {len(X)} immagini, {len(y)} target (1:1)")
    return X, y

    
def generate_submission(sample_ids, translated_embeddings, output_file="submission.csv"):
    """
    Generate a submission.csv file from translated embeddings.
    """
    print("Generating submission file...")

    if isinstance(translated_embeddings, torch.Tensor):
        translated_embeddings = translated_embeddings.cpu().numpy()

    # Create a DataFrame with sample_id and embeddings

    df_submission = pd.DataFrame({'id': sample_ids, 'embedding': translated_embeddings.tolist()})

    df_submission.to_csv(output_file, index=False, float_format='%.17g')
    print(f"✓ Saved submission to {output_file}")
    
    return df_submission


@torch.inference_mode()
def visualize_retrieval(
    pred_embeddings: torch.Tensor, 
    gt_index: int, 
    image_files: list, 
    caption_text: str, 
                       image_embeddings: torch.Tensor, k=5, dataset_path="data/train"):
    """
    Visualize a single retrieval example.
    
    Args:
        pred_embedding: (768,) single text embedding translated to image space
        gt_index: ground truth image index
        image_files: list of image filenames
        caption_text: the caption text
        image_embeddings: (N, 768) all image embeddings
        k: number of results to show
        dataset_path: path to dataset
    """    
    # Search using cosine similarity
    similarities = (image_embeddings @ pred_embeddings.T).squeeze().numpy()
    
    retrieved_indices = np.argsort(-similarities)[:k]
    distances = -similarities[retrieved_indices]
    
    # Get ground truth image name
    gt_image_name = image_files[gt_index]
    
    # Check if ground truth is in top-k
    gt_in_topk = gt_index in retrieved_indices
    gt_rank = None
    if gt_in_topk:
        gt_rank = np.where(retrieved_indices == gt_index)[0][0] + 1
    
    # Display
    fig, axes = plt.subplots(1, k + 1, figsize=(20, 4))
    
    
    # Find the correct image path
    img_path = Path(dataset_path) / "Images" / gt_image_name

    try:
        img = Image.open(img_path)
        axes[0].imshow(img)
        axes[0].set_title(f"Ground Truth\n{gt_image_name[:20]}...", fontsize=10, color='green')
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, "Image not found", ha='center', va='center')
        axes[0].axis('off')
    
    # Retrieved images
    for i, idx in enumerate(retrieved_indices):
        retrieved_name = image_files[idx]
        
        # Find the correct image path
        img_path = Path(dataset_path) / "Images" / retrieved_name
    
        
        try:
            img = Image.open(img_path)
            axes[i + 1].imshow(img)
            
            # Highlight if this is the ground truth
            color = 'green' if idx == gt_index else 'black'
            weight = 'bold' if idx == gt_index else 'normal'
            
            title = f"Rank {i+1}\nDist: {distances[i]:.2f}"
            if idx == gt_index:
                title += "\n✓ CORRECT"
            
            axes[i + 1].set_title(title, fontsize=10, color=color, weight=weight)
            axes[i + 1].axis('off')
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i + 1].axis('off')
    
    status = f"✓ Found at rank {gt_rank}" if gt_in_topk else "✗ Not in top-5"
    plt.suptitle(f"Input: '{caption_text}'", fontsize=14, weight='bold')
    # plt.suptitle(f"Text-to-Image Retrieval - {status}", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
    
    return gt_in_topk, gt_rank
        

def main():
    parser = argparse.ArgumentParser(description="Preprocess image-caption dataset and save to a .npz file.", add_help=True)
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to the dataset folder (e.g., 'data/train')."
    )
    parser.add_argument(
        "--output-file", '-o',
        type=str,
        default="processed_data.npz",
        help="Path to save the output .npz file."
    )
    parser.add_argument(
        "--create-secret-version",
        action='store_true',
        help="Create a secret version of the output file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Autodetects if not specified."
    )
    args = parser.parse_args()

    create_data_file(args.input_folder, args.output_file, args.device, args)

if __name__ == "__main__":
    main()



#################################################################### Our functions ####################################################################

import torch

def evaluate_model(model, val_loader, device, batch_size_eval=100):
    model.eval()
    total_samples = 0
    top1_total = 0
    top5_total = 0
    top10_total = 0
    all_ranks = []

    with torch.no_grad():
        for text_batch, img_batch in val_loader:
            text_batch = text_batch.to(device)
            img_batch = img_batch.to(device)

            text_emb = model(text_batch)  

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            img_batch = img_batch / img_batch.norm(dim=-1, keepdim=True)

            num_samples = text_emb.size(0)
            for start_idx in range(0, num_samples, batch_size_eval):
                end_idx = min(start_idx + batch_size_eval, num_samples)
                t_sub = text_emb[start_idx:end_idx]
                i_sub = img_batch[start_idx:end_idx]

                # Similarity by dot product
                sim = t_sub @ i_sub.T  # [B, B]

                # Compute ranks for each sample
                ranks = []
                for i in range(sim.size(0)):
                    scores = sim[i]
                    sorted_indices = torch.argsort(scores, descending=True)
                    rank = (sorted_indices == i).nonzero(as_tuple=False).item() + 1
                    ranks.append(rank)

                ranks = torch.tensor(ranks, device=device)
                all_ranks.extend(ranks.tolist())

                top1_total += (ranks == 1).sum().item()
                top5_total += (ranks <= 5).sum().item()
                top10_total += (ranks <= 10).sum().item()
                total_samples += ranks.numel()

    # Global metrics calculation
    top1_acc = top1_total / total_samples
    top5_acc = top5_total / total_samples
    top10_acc = top10_total / total_samples
    mean_rank = sum(all_ranks) / len(all_ranks)
    mrr = sum(1.0 / torch.tensor(all_ranks, dtype=torch.float)).item() / len(all_ranks)

    print(f"\nRisultati complessivi su {total_samples} campioni:")
    print(f"Top-1 accuracy : {top1_acc * 100:.2f}%")
    print(f"Top-5 accuracy : {top5_acc * 100:.2f}%")
    print(f"Top-10 accuracy: {top10_acc * 100:.2f}%")
    print(f"Rank medio     : {mean_rank:.2f}")
    print(f"MRR            : {mrr:.4f}")

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "top10_acc": top10_acc,
        "mean_rank": mean_rank,
        "mrr": mrr,
        "all_ranks": all_ranks
    }


def predict_in_batches(model, X, device, batch_size=64):

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)  # [B, 1024]
            
            out = model(batch)  
            
            out = out.cpu()                       
            preds.append(out)

    return torch.cat(preds, dim=0) 


def save_top_k_model(trial, trained_model, mrr, study, TOP_K=8, folder="models_1408_6"):
    """
    Saves the top k models based on MRR.
    """
    os.makedirs(folder, exist_ok=True)
    model_path = f"{folder}/model_1408_trial_{trial.number}.pt"

    saved_models = []
    for fname in os.listdir(folder):
        if fname.endswith(".pt"):
            try:
                tnum = int(fname.split("_")[-1].split(".")[0])
                saved_models.append((tnum, fname))
            except:
                pass

    if len(saved_models) < TOP_K:
        torch.save(trained_model.state_dict(), model_path)
        print(f"Salvato nuovo modello TOP-K (trial {trial.number}, MRR={mrr:.4f})")
        return

    saved_trial_numbers = [s[0] for s in saved_models]
    saved_trials = [t for t in study.trials if t.number in saved_trial_numbers]

    saved_trials = [t for t in saved_trials if t.value is not None]

    if not saved_trials:
        torch.save(trained_model.state_dict(), model_path)
        print(f"Salvato nuovo modello TOP-K (trial {trial.number}, MRR={mrr:.4f})")
        return

    saved_trials_sorted = sorted(saved_trials, key=lambda t: t.value, reverse=True)
    worst_saved_trial = saved_trials_sorted[-1]

    if mrr > worst_saved_trial.value:
        worst_path = f"{folder}/model_1408_trial_{worst_saved_trial.number}.pt"
        if os.path.exists(worst_path):
            os.remove(worst_path)
        torch.save(trained_model.state_dict(), model_path)
        print(f"Rimpiazzato trial {worst_saved_trial.number} (MRR={worst_saved_trial.value:.4f}) con trial {trial.number} (MRR={mrr:.4f})")
    else:
        print(f"Trial {trial.number} non entra nei Top-{TOP_K} (MRR={mrr:.4f} <= {worst_saved_trial.value:.4f})")



def objective(trial):
    global best_model
    global best_mrr
    global study

    lr = trial.suggest_float("lr", 1.5e-4, 1.75e-4, log=True)  
    temperature = trial.suggest_float("temperature", 0.12, 0.16, step=0.01)
    dropout = 0.35
    weight_decay = 3e-5
    
    model = ExpandCompressMLP(
        input_dim=1024,
        hidden_dim=1408,
        bottleneck_dim=1024,
        output_dim=1536,
        dropout=dropout,
        use_residual=False
    ).to(device)
    
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        epochs=33,
        patience=5,
        use_amp=False,
        use_scheduler=True
    )
    
    res = evaluate_model(trained_model, val_loader, device=device)
    val_loss = min(history["val_loss"])
    top1_acc = res["top1_acc"]
    mrr = res["mrr"]
    
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("top1_acc", top1_acc)
    trial.set_user_attr("mrr", mrr)
    trial.set_user_attr("params", {
        "hidden_dim": 1408,
        "lr": lr,
        "temperature": temperature,
    })
    
    save_top_k_model(trial, trained_model, mrr, study, folder="models_1408_6")
    
    global best_mrr
    if mrr > best_mrr:
        best_mrr = mrr
        best_model = trained_model
        torch.save(best_model.state_dict(), "best_model.pt")
        print(f"Salvato Best Model assoluto (MRR = {mrr:.4f})")
    
    return mrr