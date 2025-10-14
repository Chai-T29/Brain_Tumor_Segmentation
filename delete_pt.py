import os

def delete_pt_files(root_folder: str) -> None:
    """
    Recursively deletes all .pt files from the given root folder.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".pt"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    target_folder = "MU-Glioma-Post"  # adjust path if needed
    delete_pt_files(target_folder)