import pickle

def load_and_inspect_pickle(pkl_file_path):
    # 1. Load the pickle file
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Pickle file loaded successfully from {pkl_file_path}")
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return

    # 2. Inspect the structure of the loaded data
    print("\n--- Data Structure ---")
    if isinstance(data, dict):
        print(f"The data is a dictionary with keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"The data is a list with length: {len(data)}")
    else:
        print(f"Data type: {type(data)}")

    # 3. Print the first few items (for dictionaries and lists)
    if isinstance(data, dict):
        for key in data:
            print(f"\nKey: {key}")
            print(f"Type of value: {type(data[key])}")
            print(f"First 5 elements of {key}: {str(data[key])[:1000]}")
    elif isinstance(data, list):
        print("\nFirst 5 elements in the list:")
        for idx, item in enumerate(data[:5]):
            print(f"Element {idx + 1}: {str(item)[:500]}")
    else:
        print("\nData preview:")
        print(str(data)[:500])

if __name__ == '__main__':
    # Replace this with the path to your pickle file
    pkl_file_path = 'log/video_output_ori_2024-10-16-10-39-17.pkl'
    
    load_and_inspect_pickle(pkl_file_path)
