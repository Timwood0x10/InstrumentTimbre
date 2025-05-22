"""
Data Processing and Dataset Definition Module

Contains dataset classes, data loaders, and dataset splitting utilities for instrument timbre tasks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm


class ChineseInstrumentDataset(Dataset):
    """Dataset for Chinese traditional instrument timbres"""

    def __init__(
        self,
        root_dir,
        use_wav_files=False,
        augment=False,
        debug=False,
        fixed_length=256,
        feature_type="multi",
    ):
        self.root_dir = root_dir
        self.use_wav_files = use_wav_files
        self.augment = augment
        self.debug = debug
        self.fixed_length = fixed_length
        self.feature_type = feature_type
        self.samples = []

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        if self.use_wav_files:
            # Use all WAV files in the directory directly
            for file in os.listdir(self.root_dir):
                if file.endswith((".wav", ".mp3", ".flac")):
                    file_path = os.path.join(self.root_dir, file)
                    instrument = self._extract_instrument_from_filename(file)
                    if instrument:
                        self.samples.append(
                            {"path": file_path, "instrument": instrument}
                        )
        else:
            # Process by folder classification
            for instrument_folder in os.listdir(self.root_dir):
                instrument_dir = os.path.join(self.root_dir, instrument_folder)
                if os.path.isdir(instrument_dir):
                    for audio_file in os.listdir(instrument_dir):
                        if audio_file.endswith((".wav", ".mp3", ".flac")):
                            self.samples.append(
                                {
                                    "path": os.path.join(instrument_dir, audio_file),
                                    "instrument": instrument_folder,
                                }
                            )

        print(f"Loaded {len(self.samples)} audio samples")

    def _extract_instrument_from_filename(self, filename):
        """Extract instrument type from filename"""
        # Simple implementation, more complex logic might be needed in practice
        basename = os.path.splitext(filename.lower())[0]
        instruments = ["erhu", "guzheng", "pipa", "erhu", "guzheng", "pipa"]
        for instrument in instruments:
            if instrument in basename:
                return instrument
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(sample["path"], sr=22050)

        # Data augmentation
        if self.augment:
            audio = self._apply_augmentation(audio, sr)

        # Extract features
        features = self._extract_features(audio, sr)

        return features, sample["instrument"]

    def _apply_augmentation(self, audio, sr):
        """Apply data augmentation"""
        # Implement data augmentation logic
        return audio

    def _extract_features(self, audio, sr):
        """Extract audio features"""
        # Implement feature extraction logic
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Fix for tensor size inconsistency - pad or truncate to fixed length
        if self.fixed_length:
            if mel_spec.shape[1] > self.fixed_length:
                mel_spec = mel_spec[:, : self.fixed_length]
            elif mel_spec.shape[1] < self.fixed_length:
                padding = np.zeros(
                    (mel_spec.shape[0], self.fixed_length - mel_spec.shape[1])
                )
                mel_spec = np.concatenate([mel_spec, padding], axis=1)

        # Shape: [1, 128, fixed_length] - IMPORTANT: First dimension is channels (1)
        mel_spec = torch.tensor(mel_spec).float().unsqueeze(0)

        return mel_spec


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    """
    # Extract features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Convert labels to indices
    unique_labels = list(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    labels_idx = [label_to_idx[label] for label in labels]
    labels_tensor = torch.tensor(labels_idx)

    # Stack features - IMPORTANT: our model expects [batch_size, 1, height, width]
    # The first 1 is the number of channels, which should be 1 for mel spectrograms
    stacked_features = torch.stack(features)  # shape: [batch_size, 1, height, width]

    return stacked_features, labels_tensor


def prepare_dataloader(
    dataset_path,
    batch_size=32,
    num_workers=None, # Changed default to None for dynamic assignment
    use_wav_files=False,
    augment=False,
    debug=False,
):
    """Prepare standard data loader"""
    # Add path verification
    import os
    import logging
    import torch # Ensure torch is imported for cuda check

    logger = logging.getLogger(__name__)

    if not os.path.exists(dataset_path):
        logger.error(f"Data directory does not exist: {os.path.abspath(dataset_path)}")
        raise FileNotFoundError(
            f"Data directory does not exist: {os.path.abspath(dataset_path)}"
        )

    # Check if there are audio files in the directory
    audio_files_count = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):
                audio_files_count += 1

    logger.info(f"Found {audio_files_count} audio files in {dataset_path}")

    if audio_files_count == 0:
        logger.warning(f"Warning: No audio files found in {dataset_path}")
        # You can choose to continue or throw an error
        # raise ValueError(f"No audio files found in {dataset_path}")

    dataset = ChineseInstrumentDataset(
        dataset_path,
        use_wav_files=use_wav_files,
        augment=augment,
        debug=debug,
        fixed_length=256,  # Use fixed length to ensure consistent tensor sizes
    )

    logger.info(f"Dataset size: {len(dataset)} samples")

    if num_workers is None:
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
        num_workers = max(1, num_workers) # Ensure at least 1 worker
    
    pin_memory_setting = True if torch.cuda.is_available() else False
    logger.info(f"Using num_workers={num_workers} and pin_memory={pin_memory_setting} for DataLoader.")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_setting,
        collate_fn=collate_fn,  # Use custom collate function to handle variable-length sequences
    )

    return dataloader


def prepare_chinese_instrument_dataloader(
    dataset_path,
    batch_size=32,
    num_workers=None, # Changed default to None for dynamic assignment
    use_wav_files=False,
    augment=True,
    debug=False,
    feature_type="multi",
):
    """Prepare specialized data loader for Chinese traditional instruments with more features"""
    dataset = ChineseInstrumentDataset(
        dataset_path,
        use_wav_files=use_wav_files,
        augment=augment,
        debug=debug,
        feature_type=feature_type,
        fixed_length=256,  # Use fixed length to ensure consistent tensor sizes
    )

    if num_workers is None:
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
        num_workers = max(1, num_workers) # Ensure at least 1 worker

    pin_memory_setting = True if torch.cuda.is_available() else False
    # Assuming logger is defined if this function is called (it is at module level, but good practice if this were standalone)
    # logger.info(f"Using num_workers={num_workers} and pin_memory={pin_memory_setting} for Chinese Instrument DataLoader.")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_setting,
        collate_fn=collate_fn,  # Use custom collate function to handle variable-length sequences
    )

    return dataloader


def get_instrument_labels():
    """Get list of supported instrument labels"""
    return [
        "erhu",
        "guzheng",
        "pipa",
        "dizi",
        "guqin",
        "yangqin",
        "erhu",
        "guzheng",
        "pipa",
        "dizi",
        "guqin",
        "yangqin",
    ]


def get_audio_duration(file_path):
    """Get audio file duration in seconds"""
    import librosa

    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Unable to get audio duration: {file_path}, error: {e}")
        return 0


def split_audio_dataset(
    dataset_path,
    output_path,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
):
    """
    Split audio dataset into training, validation and test sets

    Parameters:
        dataset_path: Original dataset path
        output_path: Output path
        train_ratio: Training set ratio
        valid_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
    """
    import os
    import shutil
    import random
    import numpy as np

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output directories
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)

    # Traverse dataset directory
    for instrument_folder in os.listdir(dataset_path):
        instrument_dir = os.path.join(dataset_path, instrument_folder)
        if os.path.isdir(instrument_dir):
            # Create corresponding output directories
            os.makedirs(
                os.path.join(output_path, "train", instrument_folder), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_path, "valid", instrument_folder), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_path, "test", instrument_folder), exist_ok=True
            )

            # Get all audio files
            audio_files = []
            for file in os.listdir(instrument_dir):
                if file.endswith((".wav", ".mp3", ".flac")):
                    audio_files.append(file)

            # Shuffle files randomly
            random.shuffle(audio_files)

            # Calculate split points
            n_files = len(audio_files)
            train_end = int(n_files * train_ratio)
            valid_end = train_end + int(n_files * valid_ratio)

            # Split dataset
            train_files = audio_files[:train_end]
            valid_files = audio_files[train_end:valid_end]
            test_files = audio_files[valid_end:]

            # Copy files to corresponding directories
            for file in train_files:
                shutil.copy2(
                    os.path.join(instrument_dir, file),
                    os.path.join(output_path, "train", instrument_folder, file),
                )

            for file in valid_files:
                shutil.copy2(
                    os.path.join(instrument_dir, file),
                    os.path.join(output_path, "valid", instrument_folder, file),
                )

            for file in test_files:
                shutil.copy2(
                    os.path.join(instrument_dir, file),
                    os.path.join(output_path, "test", instrument_folder, file),
                )

            print(
                f"{instrument_folder}: Training set {len(train_files)}, Validation set {len(valid_files)}, Test set {len(test_files)}"
            )
