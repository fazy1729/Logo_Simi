#!/usr/bin/env python3
"""
Main script to run the logo processing pipeline:
1. extract_logos.py - Scrape logos from websites
2. compute_similarity.py - Cluster similar logos
"""
import logging
from pathlib import Path
import sys
from src.extract_logos import extract_logos_parallel
from src.compute_similarity import main as compute_similarity

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def main():
    try:
        # Step 1: Extract logos
        logging.info("Starting logo extraction...")
        extract_logos_parallel()
        logging.info("Logo extraction completed successfully")
        
        # Step 2: Compute similarities
        logging.info("Starting similarity computation...")
        compute_similarity()
        logging.info("Similarity computation completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()