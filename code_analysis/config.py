"""
Configuration settings for the RFP code analysis project.
"""
import torch

# Data file paths
DATA_FILES = [
    "../data/abstract_information.json",
    "../data/abstract_chunk.json",
    "../data/requirement_information.json",
    "../data/requirement_chunk.json",
    "../data/requirement_label.json",
    "../data/bidding_information.json",
]

# ChromaDB persistence settings
PERSIST_DIR = "./chroma_rfp"
COLLECTION = "rfp_full"

# Logging
LOG_PATH = "../realtime/rfp_results.xlsx"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
