import os
import json
from multiprocessing import Pool
from movie_coref.preprocess import preprocess_scripts as preprocess
from movie_coref.movie_coref import MovieCoreference

def process_single_script(script_file, parse_file, weights_file, hierarchical, document_len, overlap_len):
    """
    Processes a single script and parse file, performs coreference prediction, and saves the result.

    Args:
        script_file: Path to the raw script file.
        parse_file: Path to the parse file.
        weights_file: Path to the coreference model weights.
        hierarchical: Boolean for hierarchical or fusion-based mode.
        document_len: Length of the document.
        overlap_len: Overlap length for splitting.

    Returns:
        Path to the saved JSON file for the processed movie data.
    """
    movie_dir = os.path.dirname(script_file)  # Determine the movie folder
    movie_id = os.path.basename(movie_dir)  # Extract the movie ID from the folder name
    output_file = os.path.join(movie_dir, f"{movie_id}_movie_data.json")  # Output JSON file path

    if os.path.exists(output_file):
        print(f"Skipping {script_file}: already processed.")
        return output_file

    try:
        # Preprocess the script
        movie_data = preprocess([script_file], [parse_file])

        # Instantiate and run the coreference model
        movie_coref = MovieCoreference(
            preprocessed_data=movie_data,
            weights_file=weights_file,
            hierarchical=hierarchical,
            document_len=document_len,
            overlap_len=overlap_len,
            n_representative_mentions=3
        )
        movie_data = movie_coref.predict()

        # Save the result to the movie folder
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(movie_data, f, ensure_ascii=False, indent=4)
        print(f"Processed and saved: {script_file}")
        return output_file
    except Exception as e:
        print(f"Error processing {script_file}: {e}")
        return None
    
import os

def find_norm_files(pct=None):
    """
    Find all script, tags, and combined `_norm.txt` files within the directory structure.
    Optionally return a percentage of the files.

    Args:
        pct (float): A float between 0 and 1 specifying the percentage of files to return.
                     If None, return all files.

    Returns:
        script_files: List of paths to `_raw_norm.txt` files.
        tags_files: List of paths to `_tags_norm.txt` files.
        combined_files: List of paths to `_combined_norm.txt` files.
    """
    scripts_dir = '/Users/alecnaidoo/Downloads/MIDS/DATASCI_266_NLP_with_DL/W266_Project/scripts'
    script_files = []
    tags_files = []
    combined_files = []

    for imdb_id_folder in os.listdir(scripts_dir):
        movie_dir = os.path.join(scripts_dir, imdb_id_folder)
        if os.path.isdir(movie_dir):
            # Construct paths for each type of file
            raw_script_path = os.path.join(movie_dir, f'{imdb_id_folder}_raw_norm.txt')
            tags_path = os.path.join(movie_dir, f'{imdb_id_folder}_tags_norm.txt')
            combined_path = os.path.join(movie_dir, f'{imdb_id_folder}_combined_norm.txt')

            # Check for existence and add to the corresponding lists
            if os.path.exists(raw_script_path) and os.path.exists(tags_path):
                script_files.append(raw_script_path)
                tags_files.append(tags_path)
                if os.path.exists(combined_path):  # Combined is optional
                    combined_files.append(combined_path)
                else:
                    combined_files.append(None)  # Placeholder if combined file is missing

    # Limit files by percentage if `pct` is provided
    if pct is not None:
        if not (0 <= pct <= 1):
            raise ValueError("pct must be a float between 0 and 1.")
        limit = int(len(script_files) * pct)
        script_files = script_files[:limit]
        tags_files = tags_files[:limit]
        combined_files = combined_files[:limit]

    print(f"Found {len(script_files)} script files, {len(tags_files)} tags files, and {len(combined_files)} combined files.")
    return script_files, tags_files, combined_files


import os
import json
from movie_coref.preprocess import preprocess_scripts as preprocess
from movie_coref.movie_coref import MovieCoreference

def process_multiple_scripts(script_files, parse_files, weights_file, hierarchical, document_len, overlap_len):
    """
    Processes multiple scripts and parse files sequentially, performs coreference prediction, and saves the results.

    Args:
        script_files: List of paths to raw script files.
        parse_files: List of paths to parse files.
        weights_file: Path to the coreference model weights.
        hierarchical: Boolean for hierarchical or fusion-based mode.
        document_len: Length of the document.
        overlap_len: Overlap length for splitting.

    Returns:
        List of paths to the saved JSON files for the processed movie data.
    """
    output_files = []
    
    for script_file, parse_file in zip(script_files, parse_files):
        movie_dir = os.path.dirname(script_file)  # Determine the movie folder
        movie_id = os.path.basename(movie_dir)  # Extract the movie ID from the folder name
        output_file = os.path.join(movie_dir, f"{movie_id}_movie_data.json")  # Output JSON file path

        if os.path.exists(output_file):
            print(f"Skipping {script_file}: already processed.")
            output_files.append(output_file)
            continue

        try:
            # Preprocess the script
            print(f"Processing script: {script_file}")
            movie_data = preprocess([script_file], [parse_file])

            # Instantiate and run the coreference model
            movie_coref = MovieCoreference(
                preprocessed_data=movie_data,
                weights_file=weights_file,
                hierarchical=hierarchical,
                document_len=document_len,
                overlap_len=overlap_len,
                n_representative_mentions=3
            )
            movie_data = movie_coref.predict()

            # Save the result to the movie folder
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(movie_data, f, ensure_ascii=False, indent=4)
            print(f"Processed and saved: {script_file}")
            output_files.append(output_file)
        except Exception as e:
            print(f"Error processing {script_file}: {e}")
    
    return output_files

if __name__ == "__main__":
    script_files, tags_files, _ = find_norm_files(pct=1)  # Adjust the `pct` argument as needed

    weights_file = "data/Mar09_01:31:43PM_24839/movie_coref.pt"
    hierarchical = False
    document_len = 5120
    overlap_len = 2048

    processed_files = process_multiple_scripts(
        script_files, tags_files, weights_file, hierarchical, document_len, overlap_len
    )

    print(f"Processed {len(processed_files)} scripts successfully.")