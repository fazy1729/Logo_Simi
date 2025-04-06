# 🧠 Logo Similarity
**Logo Similarity** is a fast and scalable pipeline that takes logos from a `.parquet` file and clusters them based on visual similarity using OpenAI’s CLIP model. The project focuses on performance, clarity, and rich visual output — making it ideal for brand tracking, competitive analysis, or large-scale logo classification.

---

## 🚀 Project Overview

This project extracts logos from a structured dataset and groups similar ones together using state-of-the-art AI techniques.

Built in Python with an intuitive, modular design, this project offers:

- Logo extraction from `.snappy.parquet`
- Embedding generation using OpenAI's CLIP
- Dimensionality reduction (UMAP)
- Heatmaps and visualizations for interpretability
- Folder-based output for inspection
- CSV and image-based export of results

---

## ⚔️ The Struggle

This project started with one of the most challenging tasks I’ve encountered so far: **extracting and classifying logos from hundreds of websites**. It wasn’t just about coding — it was about solving real-world messiness, inconsistencies, and unpredictability.

### 🔍 Logo Extraction Challenges

**Logo extraction was extremely difficult.** Websites were built differently, logos were stored in various formats, hidden behind lazy loading or inside CSS, and rarely named intuitively. I had to iterate, debug, and rework my extraction logic multiple times to get consistent results. I explored multiple scraping techniques, including using **max_workers** for parallel processing to scrape multiple sites faster, but even that didn't fully solve the issue — there were still a lot of edge cases I had to handle, and performance optimization was crucial.

### 🧼 Classification Challenges

**Classification was another major hurdle.** Initially, I tried traditional image comparison techniques like **SSIM (Structural Similarity Index)**. However, it turned out to be **too slow**, especially when working with large datasets. Despite trying to optimize it with various configurations, SSIM wasn’t capable of handling the volume efficiently. Moreover, visually similar logos that were drastically different pixel-wise still had high SSIM scores. 

This is when I decided to pivot to more advanced methods:

💡 **CLIP, Dimensionality Reduction** — After extensive research, I turned to **CLIP (Contrastive Language-Image Pre-training)**, which allowed me to get vector representations of logos. However, even with CLIP, the journey wasn’t smooth. It required fine-tuning, experimenting with dimensionality reduction techniques (like PCA and UMAP). This approach worked significantly better, capturing the true essence of similarity between logos even when their appearances were quite different.

This wasn’t just a coding challenge — it was a **deep learning experience** in handling complex datasets, thinking like a problem solver, and pushing through technical roadblocks.

---


## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenAI CLIP** (`open_clip`)
- **FAISS** (for fast similarity search)
- **UMAP** (dimensionality reduction)
- **Matplotlib / Seaborn** (visualizations)
- **Pandas, Numpy** (data wrangling)
- **Scikit-learn** (ML utilities)

---

## 🔄 Pipeline Steps

1. **Extract logos from Parquet file**
2. **Preprocess logos** (resize, normalize)
3. **Embed each logo using CLIP**
4. **Reduce dimensions using UMAP**
5. **Generate heatmaps and cluster visualizations**
6. **Save clustered logos in folders**
7. **Export metadata to CSV and generate summaries**

---

## 🖼️ Example Outputs

| Cluster Samples | Heatmap Visualization | Logo Clustering |
|------------------|-----------------------|-----------------|
| ![clusters](/analysis_results/cluster_samples.png) | ![heatmap](/analysis_results/similarity_heatmap.png) | ![Logo Clustering](./photos_for_git/p4.png) |


## Terminal Output

When you run the program, this is what the terminal looks like:

<img src="./photos_for_git/p5.png" alt="Terminal Output Example" width="1080"/>


## ✨ Key Features

- ⚡ **Fast processing** of thousands of images
- 🧠 **Semantic understanding** using CLIP
- 🔍 **Human-readable results** via plots and folders
- 📁 **Automatic organization** of similar logos
- 💾 **Exported metadata** to CSV for analysis

---

## 📚 What I have Learned

This project gave me deep insights into:

- 🧠 How multimodal models like CLIP can be used for image similarity
- 🧮 The power of UMAP for dimensionality reduction in visual data
- 🔍 Clustering techniques like KMeans and HDBSCAN in practice
- 📊 Building meaningful visualizations to interpret machine learning results
- ⚙️ Writing efficient Python pipelines that scale to large datasets
- 🧹 Improving code robustness and modularity for future reuse

---

## 💼 Why This Matters

This project reflects my ability to:

- Combine **AI research** with **practical engineering**
- Tackle **real-world datasets** with noise and scale
- Deliver **clear visual results** useful for business and analytics
- Build pipelines that can be deployed or extended in a professional environment

Whether it's for automating quality assurance, visual deduplication, or brand tracking, this solution is both **practical and production-ready**.

---
