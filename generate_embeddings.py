# -*- coding: utf-8 -*-
"""
generate_embeddings.py

This script processes a given text file to extract unique words, finds their
corresponding GloVe word embeddings and the embeddings of their 5-character subword
tokens, performs Principal Component Analysis (PCA) to reduce dimensionality to 3D,
and saves the results along with metadata into a JSON file ('data.json').

Requirements:
- Python 3.7+
- gensim: For loading GloVe models (`pip install gensim`)
- scikit-learn: For PCA (`pip install scikit-learn`)
- numpy: For numerical operations (`pip install numpy`)
- nltk: For text tokenization (`pip install nltk`)
  - NLTK 'punkt' data: Run `nltk.download('punkt')` in Python after installing nltk,
    or the script will attempt to download it automatically if missing.

Usage:
    python generate_embeddings.py <input_text_file.txt>

Example:
    python generate_embeddings.py my_passage.txt
"""

import gensim.downloader
from sklearn.decomposition import PCA
import numpy as np
import json
import sys
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Optional, Dict, Any, Tuple, Set # For type hinting

# --- Configuration Constants ---
TOKEN_LEN: int = 5 # Length of subword tokens to generate
PCA_COMPONENTS: int = 3 # Target number of dimensions after PCA
MODEL_NAME: str = 'glove-wiki-gigaword-50' # Pre-trained GloVe model (50 dimensions)
OUTPUT_JSON: str = 'data.json' # Name of the output JSON file

# --- Helper Functions ---

def download_nltk_data_if_needed(resource: str, package: str) -> None:
    """Checks if an NLTK resource exists and downloads it if not."""
    try:
        nltk.data.find(resource)
        # print(f"NLTK resource '{resource}' found.") # Uncomment for verbose confirmation
    except nltk.downloader.DownloadError:
        print(f"NLTK resource '{resource}' ({package}) not found. Attempting download...")
        try:
            nltk.download(package)
            print(f"NLTK package '{package}' downloaded successfully.")
            # Verify after download
            nltk.data.find(resource)
        except Exception as e:
            print(f"\nError: Failed to download NLTK package '{package}'.", file=sys.stderr)
            print("Please try downloading it manually by running this in a Python interpreter:", file=sys.stderr)
            print(f">>> import nltk")
            print(f">>> nltk.download('{package}')")
            print(f"Error details: {e}", file=sys.stderr)
            sys.exit(1) # Exit if essential data is missing

def generate_tokens(word: str, length: int) -> List[str]:
    """
    Generates all unique substrings of a specific length from a word.
    Ensures the word is lowercase before processing.

    Args:
        word (str): The input word.
        length (int): The desired length of the tokens (substrings).

    Returns:
        List[str]: A list of unique tokens of the specified length.
    """
    word = str(word).lower() # Standardize to lowercase
    if len(word) < length:
        return [] # Cannot generate tokens if word is shorter than token length
    # Use a set comprehension for efficiency and uniqueness, then convert to list
    tokens = {word[i:i+length] for i in range(len(word) - length + 1)}
    return list(tokens)

def get_vector(model: Any, word: str) -> Optional[np.ndarray]:
    """
    Safely retrieves the embedding vector for a word from the loaded GloVe model.
    Handles potential KeyErrors if the word is not in the vocabulary.
    Assumes the model expects lowercase words.

    Args:
        model: The loaded gensim GloVe model object.
        word (str): The word (or token) to look up.

    Returns:
        Optional[np.ndarray]: The embedding vector as a NumPy array if found, else None.
    """
    try:
        # GloVe models are typically trained on lowercase text
        return model[word.lower()]
    except KeyError:
        # Word is not in the model's vocabulary
        return None

def process_text(text: str) -> List[str]:
    """
    Processes raw text into a sorted list of unique, lowercased alphabetic words.
    Uses NLTK for tokenization and regex for cleaning.

    Args:
        text (str): The input paragraph or text block.

    Returns:
        List[str]: A sorted list of unique words found in the text.
    """
    # Tokenize the text into words using NLTK (handles punctuation better than split)
    words: List[str] = word_tokenize(text)

    # Clean words: convert to lowercase, remove non-alphabetic characters, filter empty strings
    processed_words: Set[str] = set()
    for word in words:
        # Convert to lowercase and remove anything that isn't a letter
        cleaned_word = re.sub(r'[^a-z]', '', word.lower())
        # Add to set only if it's not empty after cleaning
        if cleaned_word:
            processed_words.add(cleaned_word)

    # Return a sorted list of the unique words
    return sorted(list(processed_words))

def perform_pca(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """
    Applies Principal Component Analysis (PCA) to reduce vector dimensions.

    Args:
        vectors (np.ndarray): A 2D NumPy array where each row is a vector.
        n_components (int): The target number of dimensions.

    Returns:
        np.ndarray: The transformed data with reduced dimensions.
    """
    print(f"Performing PCA ({vectors.shape[1]}D -> {n_components}D)...")
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    print("PCA complete.")
    return reduced_vectors

def create_embedding_entry(label: str, type: str, vector: np.ndarray) -> Dict[str, Any]:
    """
    Creates a dictionary representing a single data point for the JSON output.

    Args:
        label (str): The word or token string.
        type (str): Either 'word' or 'token'.
        vector (np.ndarray): The 3D vector (after PCA).

    Returns:
        Dict[str, Any]: A dictionary with label, type, and x, y, z coordinates.
    """
    return {
        "label": label,
        "type": type,
        "x": float(vector[0]), # Convert numpy float to standard Python float for JSON
        "y": float(vector[1]),
        "z": float(vector[2]),
    }

# --- Main Script Execution ---
def main():
    """Main function to orchestrate the embedding generation process."""

    # --- Argument Handling ---
    if len(sys.argv) != 2:
        # Provide usage instructions if the wrong number of arguments is given
        print(f"Usage: python {os.path.basename(__file__)} <input_text_file.txt>", file=sys.stderr)
        sys.exit(1) # Exit with an error code

    input_filepath: str = sys.argv[1] # Get the filepath from the command line

    # Check if the input file exists
    if not os.path.isfile(input_filepath):
        print(f"Error: Input file not found or is not a file: '{input_filepath}'", file=sys.stderr)
        sys.exit(1)

    # --- Download NLTK Data ---
    # Ensure the 'punkt' tokenizer data used by word_tokenize is available
    download_nltk_data_if_needed('tokenizers/punkt', 'punkt')

    # --- Read Input Text ---
    print(f"Reading text from: '{input_filepath}'")
    try:
        # Use 'with' statement for safe file handling (automatically closes file)
        with open(input_filepath, 'r', encoding='utf-8') as f:
            passage_text: str = f.read()
    except Exception as e:
        print(f"Error reading file '{input_filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Process Text ---
    print("Processing text to find unique words...")
    unique_words_from_passage: List[str] = process_text(passage_text)
    if not unique_words_from_passage:
        print("Warning: No valid alphabetic words found in the input text.")
        # Continue execution, but the output JSON will likely be empty.
    else:
        print(f"Found {len(unique_words_from_passage)} unique alphabetic words in passage.")

    # --- Load GloVe Model ---
    print(f"Loading GloVe model: '{MODEL_NAME}' (this may take time)...")
    try:
        # Download and load the specified pre-trained model
        model = gensim.downloader.load(MODEL_NAME)
        print("GloVe model loaded successfully.")
    except Exception as e:
        # Handle potential errors during download/loading (e.g., network issues)
        print(f"Error loading GloVe model '{MODEL_NAME}'.", file=sys.stderr)
        print("Please ensure you have an active internet connection.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Prepare Data Structures ---
    # These lists will store data aligned by index
    all_labels: List[str] = []         # Word or token strings
    all_vectors_high_dim: List[np.ndarray] = [] # Original 50D vectors
    all_types: List[str] = []          # 'word' or 'token'

    # Keep track of words and tokens actually found in the GloVe vocabulary
    valid_words_found: List[str] = []
    valid_tokens_found: Set[str] = set() # Use a set for efficient uniqueness checks

    # --- Extract Word Embeddings ---
    print("Filtering passage words against GloVe vocabulary...")
    for word in unique_words_from_passage:
        vector = get_vector(model, word)
        if vector is not None:
            # If a vector exists, store the word, its vector, and its type
            all_labels.append(word)
            all_vectors_high_dim.append(vector)
            all_types.append("word")
            valid_words_found.append(word) # Record this word as valid

    print(f"Found {len(valid_words_found)} passage words in the GloVe vocabulary.")

    # --- Generate and Extract Token Embeddings ---
    # Only generate tokens for words that were actually found in the vocabulary
    print(f"Generating and filtering {TOKEN_LEN}-char tokens for valid words...")
    for word in valid_words_found:
        tokens = generate_tokens(word, TOKEN_LEN)
        for token in tokens:
            # Check if we haven't already processed this specific token string
            if token not in valid_tokens_found:
                vector = get_vector(model, token)
                if vector is not None:
                    # If a vector exists, store the token, its vector, and its type
                    all_labels.append(token)
                    all_vectors_high_dim.append(vector)
                    all_types.append("token")
                    valid_tokens_found.add(token) # Add to set to mark as processed and valid

    print(f"Found {len(valid_tokens_found)} unique tokens (derived from valid words) in the GloVe vocabulary.")

    # --- Prepare Final Output Structure ---
    output_structure: Dict[str, Any] = {
        "passage": passage_text,
        "words": sorted(valid_words_found), # Include the list of valid words found
        "tokens": sorted(list(valid_tokens_found)), # Include the list of valid tokens found
        "embeddings": [] # Initialize empty list for embedding data
    }

    # --- Perform PCA and Finalize Output ---
    if not all_vectors_high_dim:
        # Handle the case where no valid words or tokens were found
        print("\nWarning: No valid words or tokens from the passage were found in the GloVe model.")
        print(f"Saving a structure with empty embedding data to '{OUTPUT_JSON}'.")
    else:
        # Convert list of vectors to a 2D NumPy array for PCA
        vectors_np = np.array(all_vectors_high_dim)

        # Perform PCA to get 3D vectors
        reduced_vectors: np.ndarray = perform_pca(vectors_np, PCA_COMPONENTS)

        # Create the list of embedding entries for the JSON output
        embedding_data: List[Dict[str, Any]] = []
        for i in range(len(all_labels)):
            entry = create_embedding_entry(all_labels[i], all_types[i], reduced_vectors[i])
            embedding_data.append(entry)

        # Add the embedding data to the final output structure
        output_structure["embeddings"] = embedding_data

    # --- Save Output to JSON ---
    print(f"\nSaving structured data to '{OUTPUT_JSON}'...")
    try:
        # Write the dictionary to a JSON file with pretty printing (indent=2)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_structure, f, indent=2, ensure_ascii=False)
        print(f"Successfully generated and saved data to '{OUTPUT_JSON}'.")
    except Exception as e:
        # Handle potential file writing errors
        print(f"Error writing JSON file '{OUTPUT_JSON}': {e}", file=sys.stderr)
        sys.exit(1)


# --- Entry Point Guard ---
# Ensures that the main() function is called only when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
