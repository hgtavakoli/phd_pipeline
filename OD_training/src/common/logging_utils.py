import csv
import os


def initialize_csv(file_path, header):
    """
    Initializes a CSV file with a header.
    It will write the header only if the file does not exist or is empty.

    Args:
        file_path (str): The full path to the CSV file.
        header (list): A list of strings for the header row.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write header only if the file doesn't exist or is empty (size is 0)
    # This prevents adding headers to an existing, populated file.
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

    if write_header:
        # Use 'w' mode to create the file or overwrite an empty one
        # newline='' prevents extra blank rows in the CSV
        # delimiter=';' makes the file open correctly in most versions of Excel
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(header)


def append_to_csv(file_path, data_row):
    """
    Appends a single row of data to a CSV file.

    Args:
        file_path (str): The full path to the CSV file.
        data_row (list): A list of values for the data row.
    """
    # Use 'a' mode to append to the end of the file
    # newline='' prevents extra blank rows in the CSV
    # delimiter=';' must match the one used in initialization
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(data_row)


def save_text_summary(file_path, content_dict):
    """
    Saves a dictionary of summary content to a text file.

    Args:
        file_path (str): The full path to the text file.
        content_dict (dict): A dictionary where keys are descriptions and values are the metrics.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in content_dict.items():
            f.write(f"{key}: {value}\n")