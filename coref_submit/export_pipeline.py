import os
import numpy as np
import pandas as pd
import re
import codecs
import subprocess
# from rule import parse
from rule_incl_0 import parse

# Step 1: Export raw scripts to individual text files
def export_scripts(df):
    # Create the main 'scripts' directory
    scripts_dir = 'scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    for _, row in df.iterrows():
        imdb_id = str(row['imdb_movie_id'])
        script = row['script']
        
        # Create a folder for each movie
        movie_dir = os.path.join(scripts_dir, imdb_id)
        os.makedirs(movie_dir, exist_ok=True)
        
        # Save the raw script as a text file
        raw_script_path = os.path.join(movie_dir, f'{imdb_id}_raw.txt')
        with open(raw_script_path, 'w', encoding='utf-8') as f:
            f.write(script)

    print(f"Exported raw scripts to '{scripts_dir}' directory.")

# Step 2: Process each raw script using the rule-based parser and save formatted output

def process_scripts_with_parser():
    scripts_dir = 'scripts'
    folder_count = len([name for name in os.listdir(scripts_dir) if os.path.isdir(os.path.join(scripts_dir, name))])
    print(f"Total number of folders: {folder_count}")

    script_number = 1
    for imdb_id_folder in os.listdir(scripts_dir):
        movie_dir = os.path.join(scripts_dir, imdb_id_folder)
        if os.path.isdir(movie_dir):
            raw_script_path = os.path.join(movie_dir, f'{imdb_id_folder}_raw.txt')
            rule_parser_output_path = os.path.join(movie_dir, f'{imdb_id_folder}_ruleparser.txt')
            
            # Check if the raw script file exists
            if os.path.exists(raw_script_path):
                # Use the parse function from rule.py to process and save the parsed output
                parse(
                    file_orig=raw_script_path,
                    save_dir=movie_dir,
                    abr_flag='off',
                    tag_flag='on',  # Set to 'on' if you need to save tags
                    char_flag='off'
                )
                print(f"script #{script_number} ({imdb_id_folder}) processed")
                script_number += 1

    print("Processed and saved parsed script files.")

# Step 3: Export raw scripts to individual text files
def export_ml_scripts(df):
    # Create the main 'scripts' directory
    scripts_dir = 'scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    for _, row in df.iterrows():
        imdb_id = str(row['imdb_movie_id'])
        script = row['script']
        
        # Create a folder for each movie
        movie_dir = os.path.join(scripts_dir, imdb_id)
        os.makedirs(movie_dir, exist_ok=True)
        
        # Save the raw script as a text file
        raw_script_path = os.path.join(movie_dir, f'{imdb_id}_ml.txt')
        with open(raw_script_path, 'w', encoding='utf-8') as f:
            f.write(script)

    print(f"Exported raw scripts to '{scripts_dir}' directory.")

import os

def combine_scripts_with_tags():
    scripts_dir = 'scripts'
    folder_count = len([name for name in os.listdir(scripts_dir) if os.path.isdir(os.path.join(scripts_dir, name))])
    print(f"Total number of folders: {folder_count}")

    script_number = 1
    for imdb_id_folder in os.listdir(scripts_dir):
        movie_dir = os.path.join(scripts_dir, imdb_id_folder)
        if os.path.isdir(movie_dir):
            raw_script_path = os.path.join(movie_dir, f'{imdb_id_folder}_raw.txt')
            tags_path = os.path.join(movie_dir, f'{imdb_id_folder}_raw_tags.txt')
            combined_output_path = os.path.join(movie_dir, f'{imdb_id_folder}_combined.txt')

            if os.path.exists(raw_script_path) and os.path.exists(tags_path):
                
                # Read lines
                with open(raw_script_path, 'r', encoding='utf-8') as raw_file:
                    raw_lines = raw_file.readlines()

                with open(tags_path, 'r', encoding='utf-8') as tags_file:
                    tag_lines = tags_file.readlines()

                # Check line counts
                if len(raw_lines) != len(tag_lines):
                    print(f"Error: Mismatched line counts in {imdb_id_folder} - Raw: {len(raw_lines)}, Tags: {len(tag_lines)}. Skipping.")
                    continue

                # Combine lines
                with open(combined_output_path, 'w', encoding='utf-8') as combined_file:
                    for tag_line, raw_line in zip(tag_lines, raw_lines):
                        combined_file.write(f"{tag_line.strip()}: {raw_line.strip()}\n")

                print(f"script #{script_number} ({imdb_id_folder}) combined and saved.")
                script_number += 1

    print("Processed and saved combined script files.")

# Step 5 : normalize for excessive spaces
def normalize_combined_and_raw(combined_file: str, raw_file: str, raw_tags_file: str, output_combined_file: str, output_raw_file: str, output_raw_tags_file: str):
    """
    Normalize combined and raw text files by removing '0'-tagged lines from the combined file and
    the corresponding lines from the raw file.

    Args:
        combined_file (str): Path to the combined file (tags and text).
        raw_file (str): Path to the raw script file.
        output_combined_file (str): Path to save the normalized combined file.
        output_raw_file (str): Path to save the normalized raw file.
    """
    # Read the combined file
    with open(combined_file, "r") as f:
        combined_lines = f.readlines()

    # Read the raw file
    with open(raw_file, "r") as f:
        raw_lines = f.readlines()

    # Read the raw tags file
    with open(raw_tags_file, "r") as f:
        raw_tag_lines = f.readlines()

    # Ensure the combined and raw files have the same number of lines
    assert len(combined_lines) == len(raw_lines), (
        f"Mismatch in line counts: combined ({len(combined_lines)}), raw ({len(raw_lines)})"
    )

    # Initialize lists for filtered combined and raw lines
    filtered_combined_lines = []
    filtered_raw_lines = []
    filtered_raw_tags_lines = []

    # Process each line in the combined file
    for idx, line in enumerate(combined_lines):
        # Split into tag and text parts (assume format "TAG: text")
        if ':' in line:
            tag, text = line.split(':', 1)
            tag = tag.strip()
        else:
            tag, text = line.strip(), ""

        # Keep only lines where the tag is not '0'
        if tag != '0':
            filtered_combined_lines.append(line)
            filtered_raw_lines.append(raw_lines[idx])
            filtered_raw_tags_lines.append(raw_tag_lines[idx])

    # Write the filtered lines to the output files
    with open(output_combined_file, "w") as f:
        f.writelines(filtered_combined_lines)

    with open(output_raw_file, "w") as f:
        f.writelines(filtered_raw_lines)

    with open(output_raw_tags_file, "w") as f:
        f.writelines(filtered_raw_tags_lines)

    print(f"Normalized files saved to:\n  - Combined: {output_combined_file}\n  - Raw: {output_raw_file}")

def normalize_all_scripts():
    """
    Normalize all scripts in the given directory structure by processing each movie's 
    _combined.txt and _raw.txt files.
    """

    scripts_dir = "scripts" 
    # Iterate through all subfolders in the scripts directory
    for imdb_id_folder in os.listdir(scripts_dir):
        movie_dir = os.path.join(scripts_dir, imdb_id_folder)
        
        if os.path.isdir(movie_dir):
            # Define paths for the combined and raw files
            combined_file = os.path.join(movie_dir, f'{imdb_id_folder}_combined.txt')
            raw_file = os.path.join(movie_dir, f'{imdb_id_folder}_raw.txt')
            raw_tags_file = os.path.join(movie_dir, f'{imdb_id_folder}_raw_tags.txt')

            # Define output paths for normalized files
            output_combined_file = os.path.join(movie_dir, f'{imdb_id_folder}_combined_norm.txt')
            output_raw_file = os.path.join(movie_dir, f'{imdb_id_folder}_raw_norm.txt')
            output_raw_tags_file = os.path.join(movie_dir, f'{imdb_id_folder}_tags_norm.txt')

            # Check if both files exist
            if os.path.exists(combined_file) and os.path.exists(raw_file) and os.path.exists(raw_tags_file):
                try:
                    # Normalize the files
                    normalize_combined_and_raw(combined_file, raw_file, raw_tags_file, output_combined_file, output_raw_file, output_raw_tags_file)
                    print(f"Normalized scripts for movie {imdb_id_folder} saved.")
                except AssertionError as e:
                    print(f"Error normalizing scripts for movie {imdb_id_folder}: {e}")
            else:
                print(f"Files missing for movie {imdb_id_folder}. Skipping...")
    print("Normalization completed for all movies.")


if __name__ == "__main__":
    # Load the dataframe from the CSV file
    df = pd.read_csv(r'/Users/alecnaidoo/Downloads/MIDS/DATASCI_266_NLP_with_DL/W266_Project/mica-character-coref/poetry-mica-char-coref/robust_parser.csv')
    # Rename the first column to 'imdb_movie_id'
    df.rename(columns={'0': 'imdb_movie_id'}, inplace=True)
    # Filter the dataframe to include only rows with 'imdb_movie_id' and 'script' columns, and drop rows with missing 'script' values
    df = df[['imdb_movie_id', 'script']].dropna(subset=['script'])
    # Export the scripts to individual text files
    # export_scripts(df)
    # Process the exported scripts using the rule-based parser
    # process_scripts_with_parser()
    # 3 ml scripts
    # export_ml_scripts(df)
    # 4 combine raw and tag files
    # combine_scripts_with_tags()
    # 5 normalize scripts
    normalize_all_scripts()