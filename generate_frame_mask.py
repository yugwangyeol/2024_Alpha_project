import csv
import numpy as np
import os

def create_npy_from_csv(csv_file_path, output_directory):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        current_val = None
        current_rec = None
        data = None

        for row in csv_reader:
            if not row or len(row) < 5:  # Skip empty rows or invalid rows
                continue

            val, rec, start, end, label = row[0], row[1], row[2], row[3], row[4]

            if val:
                if current_val and current_rec and data is not None:
                    save_npy(current_val, current_rec, data, output_directory)
                    data = None
                current_val = val
            if rec:
                if current_rec and data is not None:
                    save_npy(current_val, current_rec, data, output_directory)
                    data = None
                current_rec = rec

            if current_val and current_rec:
                if data is None:
                    data = np.zeros(10000, dtype=np.int8)

                # Skip rows with empty start or end values
                if start and end:
                    try:
                        start = int(start)
                        end = int(end)
                        value = 1 if label == 'A' else 0
                        data[start:end+1] = value
                    except ValueError as e:
                        print(f"Error processing row: {row}. Error: {e}")

        # Save the last dataset
        if data is not None:
            save_npy(current_val, current_rec, data, output_directory)

def save_npy(val, rec, data, output_directory):
    val_num = int(val[3:])
    rec_num = int(rec[3:])
    filename = f"{val_num:02d}_{rec_num:04d}.npy"
    filepath = os.path.join(output_directory, filename)
    np.save(filepath, data)
    print(f"Saved {filepath}")

# Usage
csv_file_path = 'DAD/LABEL.csv'
output_directory = 'DAD_Jigsaw/testing/frame_masks'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

create_npy_from_csv(csv_file_path, output_directory)