import os
import clip
import torch
import numpy as np
import hashlib
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
import plotly.express as px
from pathlib import Path

# ---------------------
# CONFIGURATION
# ---------------------
class Config:
    LOGO_FOLDER = "./logos"
    OUTPUT_FOLDER = "./output_clusters"
    ANALYSIS_FOLDER = "./analysis_results"
    SIMILARITY_THRESHOLD = 0.80  # 80% similarity (We can change this number for better accuracy)
    USE_HASH_CHECK = True
    MAX_CLUSTER_DISPLAY = 5
    SAMPLE_SIZE = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "ViT-B/32"
    RANDOM_SEED = 42
    CLUSTER_PLOT_SIZE = (15, 15)
    HEATMAP_SIZE = (20, 15)
    BATCH_SIZE = 32  # Process images in batches to avoid memory issues

# ---------------------
# IMAGE PROCESSING
# ---------------------
class LogoProcessor:
    def __init__(self):
        self.device = Config.DEVICE
        self.model, self.preprocess = clip.load(Config.MODEL_NAME, device=self.device)
        self.image_paths = []
        self.embeddings = []
        self.hashes = []
        self.clusters = []
        self.cluster_stats = defaultdict(dict)
        self.similarity_matrix = None

    def load_images(self):
        self.image_paths = [
            os.path.join(Config.LOGO_FOLDER, f) 
            for f in os.listdir(Config.LOGO_FOLDER) 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        
        if Config.SAMPLE_SIZE:
            np.random.seed(Config.RANDOM_SEED)
            self.image_paths = list(np.random.choice(self.image_paths, Config.SAMPLE_SIZE, replace=False))

    def compute_embeddings(self):
        print(f"Computing embeddings for {len(self.image_paths)} images...")
        
        # Process images in batches
        for i in tqdm(range(0, len(self.image_paths), Config.BATCH_SIZE), desc="Processing images"):
            batch_paths = self.image_paths[i:i+Config.BATCH_SIZE]
            batch_images = []
            
            for path in batch_paths:
                if Config.USE_HASH_CHECK:
                    with open(path, "rb") as f:
                        self.hashes.append(hashlib.md5(f.read()).hexdigest())
                
                try:
                    image = self.preprocess(Image.open(path).convert("RGB"))
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    batch_images.append(None)
            
            # Filter out failed images
            valid_indices = [j for j, img in enumerate(batch_images) if img is not None]
            valid_images = [batch_images[j] for j in valid_indices]
            
            if valid_images:
                images_tensor = torch.stack(valid_images).to(self.device)
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(images_tensor).cpu().numpy()
                
                # Store embeddings with proper indexing
                for j, idx in enumerate(valid_indices):
                    self.embeddings.append(batch_embeddings[j])
        
        self.embeddings = np.array(self.embeddings)

    def compute_similarity_matrix(self):
        print("Computing similarity matrix...")
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings available for similarity computation")
        
        # Normalize embeddings first for more efficient computation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / norms
        
        # Compute similarity matrix in chunks if large
        if len(normalized_embeddings) > 1000:
            print("Large dataset detected, computing similarity in chunks...")
            chunk_size = 500
            similarity_matrix = np.zeros((len(normalized_embeddings), len(normalized_embeddings)))
            
            for i in tqdm(range(0, len(normalized_embeddings), chunk_size), desc="Computing chunks"):
                for j in range(0, len(normalized_embeddings), chunk_size):
                    chunk_sim = np.dot(
                        normalized_embeddings[i:i+chunk_size],
                        normalized_embeddings[j:j+chunk_size].T
                    )
                    similarity_matrix[i:i+chunk_size, j:j+chunk_size] = chunk_sim
        else:
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        self.similarity_matrix = similarity_matrix
        print("Similarity matrix computed successfully")

    def create_strict_clusters(self):
        print("Creating strict similarity clusters...")
        
        # Create adjacency matrix
        adj_matrix = self.similarity_matrix >= Config.SIMILARITY_THRESHOLD
        n = len(self.image_paths)
        visited = np.zeros(n, dtype=bool)
        self.clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                
                queue = list(np.where(adj_matrix[i])[0])
                while queue:
                    j = queue.pop()
                    if not visited[j] and np.all(adj_matrix[j][cluster]):
                        cluster.append(j)
                        visited[j] = True
                        queue.extend([x for x in np.where(adj_matrix[j])[0] if not visited[x]])
                
                self.clusters.append([self.image_paths[idx] for idx in cluster])
                
                cluster_sims = []
                for x in range(len(cluster)):
                    for y in range(x+1, len(cluster)):
                        cluster_sims.append(self.similarity_matrix[cluster[x]][cluster[y]])
                
                cluster_id = len(self.clusters)
                self.cluster_stats[f"cluster_{cluster_id}"] = {
                    "size": len(cluster),
                    "sample_files": [os.path.basename(self.image_paths[idx]) for idx in cluster[:3]],
                    "average_similarity": float(np.mean(cluster_sims)) if cluster_sims else 0.0,
                    "min_similarity": float(np.min(cluster_sims)) if cluster_sims else 0.0
                }
        
        self.clusters.sort(key=len, reverse=True)
        print(f"Created {len(self.clusters)} clusters with strict mutual similarity")

    def save_clusters(self):
        """Save clusters with proper filenames"""
        print(f"Saving {len(self.clusters)} clusters...")
        
        # Clear output directory
        shutil.rmtree(Config.OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        
        # Save each cluster
        for cluster_idx, cluster in enumerate(self.clusters, 1):
            cluster_dir = os.path.join(Config.OUTPUT_FOLDER, f"cluster_{cluster_idx}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            for img_path in cluster:
                try:
                    shutil.copy2(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))
                except Exception as e:
                    print(f"Error copying {img_path}: {e}")
        
        # Save metadata
        with open(os.path.join(Config.ANALYSIS_FOLDER, "cluster_metadata.json"), "w") as f:
            json.dump(self.cluster_stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else x)

    def visualize_results(self):
        print("Generating visualizations...")
        
        plt.figure(figsize=(10, 6))
        sns.histplot([len(c) for c in self.clusters], bins=20, kde=True)
        plt.title("Cluster Size Distribution")
        plt.xlabel("Number of images")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(Config.ANALYSIS_FOLDER, "cluster_sizes.png"))
        plt.close()
        
        # Sample heatmap
        sample_size = min(50, len(self.image_paths))
        sample_indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
        plt.figure(figsize=Config.HEATMAP_SIZE)
        sns.heatmap(
            self.similarity_matrix[np.ix_(sample_indices, sample_indices)],
            cmap="YlOrRd",
            xticklabels=[os.path.basename(self.image_paths[i])[:15] for i in sample_indices],
            yticklabels=[os.path.basename(self.image_paths[i])[:15] for i in sample_indices]
        )
        plt.title("Similarity Heatmap (Sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.ANALYSIS_FOLDER, "similarity_heatmap.png"))
        plt.close()
        
        # Sample clusters visualization
        self._display_sample_clusters()

    def _display_sample_clusters(self):
        plt.figure(figsize=Config.CLUSTER_PLOT_SIZE)
        clusters_to_display = min(10, len(self.clusters))
        
        for cluster_idx, cluster in enumerate(self.clusters[:clusters_to_display]):
            for img_idx, img_path in enumerate(cluster[:Config.MAX_CLUSTER_DISPLAY]):
                try:
                    img = Image.open(img_path)
                    plt.subplot(
                        clusters_to_display,
                        Config.MAX_CLUSTER_DISPLAY,
                        cluster_idx * Config.MAX_CLUSTER_DISPLAY + img_idx + 1
                    )
                    plt.imshow(img)
                    plt.axis("off")
                    if img_idx == 0:
                        plt.title(f"Cluster {cluster_idx+1}\n({len(cluster)} images)")
                except Exception as e:
                    print(f"Error displaying {img_path}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.ANALYSIS_FOLDER, "cluster_samples.png"))
        plt.close()

# ---------------------
# MAIN EXECUTION
# ---------------------
def main():
    # Setup folders
    Path(Config.OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(Config.ANALYSIS_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Initialize and process
    processor = LogoProcessor()
    processor.load_images()
    processor.compute_embeddings()
    processor.compute_similarity_matrix()
    processor.create_strict_clusters()
    processor.save_clusters()
    processor.visualize_results()
    
    print(f"Processing complete! Results saved to {Config.OUTPUT_FOLDER} and {Config.ANALYSIS_FOLDER}")

if __name__ == "__main__":
    main()