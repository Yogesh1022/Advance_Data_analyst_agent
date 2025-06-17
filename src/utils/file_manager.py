import os
import shutil
import io

class FileManager:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def save_uploaded_file(self, file_content: io.BytesIO, file_name: str, destination_dir: str = "raw") -> str:
        """
        Saves an uploaded file (BytesIO) to the specified data directory.
        Returns the full path to the saved file.
        """
        if destination_dir == "raw":
            target_dir = self.raw_data_dir
        elif destination_dir == "processed":
            target_dir = self.processed_data_dir
        else:
            raise ValueError("destination_dir must be 'raw' or 'processed'")

        destination_path = os.path.join(target_dir, file_name)

        # Write BytesIO content to a file
        with open(destination_path, "wb") as f:
            f.write(file_content.getvalue())
        
        return destination_path

    def get_file_extension(self, file_name: str) -> str:
        """Returns the lowercase extension of a file."""
        return os.path.splitext(file_name)[1].lower()

    def get_raw_file_path(self, file_name: str) -> str:
        """Returns the full path for a file in the raw data directory."""
        return os.path.join(self.raw_data_dir, file_name)

    def get_processed_file_path(self, file_name: str) -> str:
        """Returns the full path for a file in the processed data directory."""
        return os.path.join(self.processed_data_dir, file_name)

    def delete_file(self, file_path: str):
        """Deletes a file."""
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"File not found for deletion: {file_path}")