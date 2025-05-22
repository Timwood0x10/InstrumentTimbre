"""
Instrument Timbre Model - Definition for Instrument Timbre Analysis and Conversion Model

Includes the main model class, feature extraction, timbre application, and other core functionalities.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import time
from pathlib import Path
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import librosa
from torch.cuda.amp import GradScaler, autocast # For AMP

try:
    from .encoders import InstrumentTimbreEncoder, ChineseInstrumentTimbreEncoder
    from .decoders import InstrumentTimbreDecoder, EnhancedTimbreDecoder
    from ..utils.cache import FeatureCache
    from ..audio.processors import (
        load_audio,
        save_audio,
        extract_features,
        extract_chinese_instrument_features,
    )
except ImportError:
    from models.encoders import InstrumentTimbreEncoder, ChineseInstrumentTimbreEncoder
    from models.decoders import InstrumentTimbreDecoder, EnhancedTimbreDecoder
    from utils.cache import FeatureCache
    from audio.processors import (
        load_audio,
        save_audio,
        extract_features,
        extract_chinese_instrument_features,
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SourceSeparator:
    """
    Handles audio source separation using Demucs.
    """
    def __init__(self, device):
        self.device = device
        self._demucs_model = None
        self._apply_demucs = None

    def _load_demucs_model(self):
        """
        Lazy load the Demucs model for source separation.
        """
        if self._demucs_model is None:
            try:
                # Import here to keep Demucs as an optional dependency for the class
                import torch 
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                logger.info("Loading Demucs model for source separation...")
                self._demucs_model = get_model("htdemucs")
                self._demucs_model.to(self.device)
                logger.info("Demucs model loaded")

                # Save reference to apply function
                self._apply_demucs = apply_model
            except ImportError as e:
                logger.error(f"Could not import demucs: {e}")
                logger.error("Please install demucs with: pip install demucs")
                return False
            except Exception as e:
                logger.error(f"Error loading Demucs model: {e}")
                return False
        return True

    def separate_audio_sources(self, audio_file, output_dir=None):
        """
        Separate audio into different instrument sources using Demucs.

        Args:
            audio_file: Path to input audio file.
            output_dir: Directory to save separated tracks.

        Returns:
            Dictionary with paths to separated sources, or None on failure.
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(audio_file), "separated")
        os.makedirs(output_dir, exist_ok=True)

        if not self._load_demucs_model():
            return None

        try:
            import torch # Already imported for _load_demucs_model, but good for clarity
            import torchaudio # Assuming torchaudio is available for audio loading

            logger.info(f"Loading audio file: {audio_file}")
            audio, sr = torchaudio.load(audio_file)

            if audio.shape[0] > 2: # More than stereo
                logger.warning(f"Audio has {audio.shape[0]} channels, using first two only.")
                audio = audio[:2]
            elif audio.shape[0] == 1: # Mono
                audio = audio.repeat(2, 1) # Duplicate mono to stereo for Demucs

            if sr != 44100: # Demucs expects 44.1 kHz
                logger.info(f"Resampling from {sr} to 44100 Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(self.device)
                audio = resampler(audio.to(self.device)) # Move to device before resampling if not already
            else:
                audio = audio.to(self.device)
            
            logger.info("Separating audio sources...")
            sources = self._apply_demucs(self._demucs_model, audio)
            source_names = self._demucs_model.sources

            output_files = {}
            for i, name in enumerate(source_names):
                source_audio = sources[i]
                source_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}_{name}.wav"
                source_path = os.path.join(output_dir, source_filename)
                
                torchaudio.save(source_path, source_audio.cpu(), 44100) # Save at 44.1kHz
                output_files[name] = source_path
                logger.info(f"Saved {name} track to {source_path}")

            return {"sources": output_files, "source_names": source_names}

        except Exception as e:
            logger.error(f"Error separating audio sources: {e}", exc_info=True)
            return None


class ModelPersistenceManager:
    """
    Handles loading and saving of model checkpoints for InstrumentTimbreModel.
    """
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _load_partial_weights_in_manager(self, checkpoint):
        """Load partial weights in compatibility mode, adapted for manager context."""
        # Attempt to load partial weights using manual parameter mapping
        if "encoder" in checkpoint:
            encoder_state = checkpoint["encoder"]

            # Manually load shared layers
            own_state = self.encoder.state_dict()
            for name, param in encoder_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    logger.info(f"Loaded parameter: {name}")

            # Special handling for attention mechanism layers
            if (
                "attention.gamma" in encoder_state
                and "self_attention.gamma" in own_state
            ):
                own_state["self_attention.gamma"].copy_(
                    encoder_state["attention.gamma"]
                )
                logger.info("Mapped attention.gamma -> self_attention.gamma")

            if (
                "attention.query.weight" in encoder_state
                and "self_attention.query.weight" in own_state
            ):
                own_state["self_attention.query.weight"].copy_(
                    encoder_state["attention.query.weight"]
                )
                own_state["self_attention.query.bias"].copy_(
                    encoder_state["attention.query.bias"]
                )
                own_state["self_attention.key.weight"].copy_(
                    encoder_state["attention.key.weight"]
                )
                own_state["self_attention.key.bias"].copy_(
                    encoder_state["attention.key.bias"]
                )
                own_state["self_attention.value.weight"].copy_(
                    encoder_state["attention.value.weight"]
                )
                own_state["self_attention.value.bias"].copy_(
                    encoder_state["attention.value.bias"]
                )
                logger.info("Mapped attention mechanism parameters")

        # Load decoder weights
        if "decoder" in checkpoint:
            decoder_state = checkpoint["decoder"]
            own_state = self.decoder.state_dict()
            for name, param in decoder_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    logger.info(f"Loaded decoder parameter: {name}")

        logger.info("Partial model weights loaded in compatibility mode")

    def load_model(self, model_path):
        """
        Load a saved model from model_path.

        Args:
            model_path: Path to the saved model.

        Returns:
            A dictionary with loading status and information:
            {
                'success': bool,
                'config': dict (loaded model config, empty if not found),
                'compat_mode': bool (True if compatibility mode was used)
            }
        """
        compat_mode_used = False
        loaded_config = {}
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            try:
                # Attempt to load normally
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.decoder.load_state_dict(checkpoint["decoder"])
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                # If normal loading fails, enable compatibility mode
                logger.error(f"Error loading model from {model_path}: {e}")
                logger.info("Enabling compatibility mode for older model format")
                compat_mode_used = True
                # Attempt to load partial weights
                self._load_partial_weights_in_manager(checkpoint)

            # Load configuration if exists
            if "config" in checkpoint:
                loaded_config = checkpoint["config"]
                logger.info(f"Loaded model configuration: {loaded_config}")
            
            return {'success': True, 'config': loaded_config, 'compat_mode': compat_mode_used}

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return {'success': False, 'config': {}, 'compat_mode': False}

    def save_model(self, save_path, current_config_dict):
        """
        Save the model.

        Args:
            save_path: Path to save the model.
            current_config_dict: Dictionary containing current configuration to save (e.g., chinese_instruments).
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        try:
            # Save encoder and decoder state dictionaries
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "config": current_config_dict,
            }

            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}")
            return False


# Custom Chroma transform implementation since torchaudio.transforms.Chroma is not available
class ChromaTransform:
    """Custom implementation of Chroma transform using librosa"""

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_chroma=12):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def __call__(self, audio):
        """Convert audio tensor to chroma features"""
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
            # If multi-channel, take the first channel
            if audio_np.ndim > 1 and audio_np.shape[0] > 1:
                audio_np = audio_np[0]
        else:
            audio_np = audio

        # Extract chroma using librosa
        chroma = librosa.feature.chroma_stft(
            y=audio_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )

        # Convert back to torch tensor
        return torch.from_numpy(chroma).float()


class AudioFeatureProcessor:
    """
    Handles audio feature extraction processes like Mel spectrogram, Chroma, and MFCC.
    """

    def __init__(self, feature_cache=None, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128, n_chroma=12, n_mfcc=20, device=None):
        self.feature_cache = feature_cache
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.device = device if device else torch.device("cpu") # Default to CPU if no device specified

        # Mel spectrogram converter
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        ).to(self.device)

        # Chroma converter - using custom implementation
        self.chroma_transform = ChromaTransform(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_chroma=self.n_chroma
        ) # ChromaTransform itself handles numpy conversion, so device placement is for input tensor

        # MFCC converter
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": self.n_mels},
        ).to(self.device)

    def to_mel_spectrogram(self, audio, sample_rate=None):
        """
        Convert audio to Mel spectrogram

        Args:
            audio: Audio data Tensor [channels, samples]
            sample_rate: Sample rate of the input audio. If None, assumes self.sample_rate.

        Returns:
            Mel spectrogram Tensor [n_mels, time]
        """
        current_sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        # Ensure correct sample rate
        if current_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_sample_rate, new_freq=self.sample_rate
            ).to(audio.device)
            audio = resampler(audio)

        # Extract Mel spectrogram
        mel_spec = self.mel_transform(audio.to(self.device))

        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec

    def to_chroma(self, audio, sample_rate=None):
        """
        Convert audio to chroma features

        Args:
            audio: Audio data Tensor [channels, samples] or path to audio file for caching.
            sample_rate: Sample rate of the input audio. If None, assumes self.sample_rate.

        Returns:
            Chroma features Tensor [n_chroma, time]
        """
        current_sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        # Check cache first if available and audio is a file path
        if (
            self.feature_cache # Check if feature_cache is enabled
            and isinstance(audio, str)
            and os.path.exists(audio)
        ):
            cached_chroma = self.feature_cache.get_chroma(audio)
            if cached_chroma is not None:
                return cached_chroma.to(self.device)


        # Ensure audio is a tensor for processing
        if isinstance(audio, str): # If it was a path but not in cache, load it
            loaded_audio, loaded_sr = load_audio(audio, sr=current_sample_rate, mono=True) # Assuming load_audio is available
            if loaded_audio is None:
                # Fallback for safety, though ideally load_audio would raise an error or be checked by caller
                return torch.zeros((self.n_chroma, 128), device=self.device) # Default fallback size
            audio_tensor = loaded_audio
            current_sample_rate = loaded_sr
        else:
            audio_tensor = audio


        # Ensure correct sample rate
        if current_sample_rate != self.sample_rate and isinstance(audio_tensor, torch.Tensor):
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_sample_rate, new_freq=self.sample_rate
            ).to(audio_tensor.device)
            audio_tensor = resampler(audio_tensor)

        # Extract chroma features
        try:
            # ChromaTransform expects CPU tensor or numpy array
            chroma = self.chroma_transform(audio_tensor.cpu()) 
            chroma = chroma.to(self.device)


            # Handle NaN values
            if isinstance(chroma, torch.Tensor) and torch.isnan(chroma).any():
                mel_shape_fallback_dim = 128 # Default fallback time dimension
                if isinstance(audio_tensor, torch.Tensor):
                    try:
                        # Attempt to get a more reasonable shape based on potential mel_spec
                        # This is a bit indirect; ideally, the time dimension should be calculated from input audio length
                        mel_spec_temp = self.mel_transform(audio_tensor.to(self.device))
                        mel_shape_fallback_dim = mel_spec_temp.shape[-1]
                    except Exception:
                        pass # Use default if mel_transform fails
                chroma = torch.zeros((self.n_chroma, mel_shape_fallback_dim), device=self.device)


            # Cache the result if it was a file path
            if (
                self.feature_cache # Check if feature_cache is enabled
                and isinstance(audio, str) # Check original audio type
                and os.path.exists(audio)
                and isinstance(chroma, torch.Tensor)
            ):
                self.feature_cache.put_chroma(audio, chroma.cpu()) # Cache CPU tensor

            return chroma

        except Exception as e:
            logger.error(f"Chroma extraction failed: {e}")
            mel_shape_fallback_dim = 128 # Default fallback time dimension
            if isinstance(audio_tensor, torch.Tensor):
                try:
                    mel_spec_temp = self.mel_transform(audio_tensor.to(self.device))
                    mel_shape_fallback_dim = mel_spec_temp.shape[-1]
                except Exception:
                    pass
            return torch.zeros((self.n_chroma, mel_shape_fallback_dim), device=self.device)

    def to_mfcc(self, audio, sample_rate=None):
        """
        Convert audio to MFCC features

        Args:
            audio: Audio data Tensor [channels, samples]
            sample_rate: Sample rate of the input audio. If None, assumes self.sample_rate.

        Returns:
            MFCC features Tensor [n_mfcc, time]
        """
        current_sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        # Ensure correct sample rate
        if current_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_sample_rate, new_freq=self.sample_rate
            ).to(audio.device)
            audio = resampler(audio)

        # Extract MFCC
        try:
            mfcc = self.mfcc_transform(audio.to(self.device))

            # Handle NaN values
            if torch.isnan(mfcc).any():
                # Fallback: use the shape from mel_transform as a guess for time dimension
                time_dim = self.mel_transform(audio.to(self.device)).shape[-1]
                mfcc = torch.zeros((self.n_mfcc, time_dim), device=self.device)
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            # Fallback: use the shape from mel_transform as a guess for time dimension
            time_dim = 128 # Default fallback
            try:
                time_dim = self.mel_transform(audio.to(self.device)).shape[-1]
            except Exception:
                pass
            mfcc = torch.zeros((self.n_mfcc, time_dim), device=self.device)

        return mfcc


class InstrumentTimbreModel:
    """
    Main class for instrument timbre analysis and manipulation
    """

    def __init__(
        self,
        model_path=None,
        use_pretrained=False,
        chinese_instruments=False,
        feature_caching=True,
        cache_dir=None,
        device=None,
    ):
        """
        Initialize the InstrumentTimbreModel

        Args:
            model_path: Path to load a saved model
            use_pretrained: Whether to use pretrained audio models
            chinese_instruments: Whether to use specialized Chinese instrument models
            feature_caching: Whether to use feature caching
            cache_dir: Directory for feature cache
            device: Device to use for computation (None for auto-detection)
        """
        self.chinese_instruments = chinese_instruments

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize feature cache if enabled
        self.feature_caching = feature_caching
        self.feature_cache = None # Initialize to None
        if feature_caching:
            self.feature_cache = FeatureCache(cache_dir)
            logger.info(f"Feature cache initialized at {self.feature_cache.cache_dir}")

        # Initialize AudioFeatureProcessor
        self.feature_processor = AudioFeatureProcessor(feature_cache=self.feature_cache, device=self.device)

        # Initialize encoder based on configuration
        if chinese_instruments:
            self.encoder = ChineseInstrumentTimbreEncoder(
                feature_channels=2, use_wavelet=True
            )
            logger.info("Using specialized Chinese instrument encoder")
        else:
            self.encoder = InstrumentTimbreEncoder(use_pretrained=use_pretrained)
            logger.info(f"Using standard encoder with pretrained: {use_pretrained}")

        # Initialize decoder
        if chinese_instruments:
            self.decoder = EnhancedTimbreDecoder(feature_dim=128, with_residual=True)
            logger.info("Using enhanced decoder with residual connections")
        else:
            self.decoder = InstrumentTimbreDecoder()
            logger.info("Using standard decoder")

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Initialize ModelPersistenceManager
        self.persistence_manager = ModelPersistenceManager(self.encoder, self.decoder, self.device)

        # Initialize SourceSeparator
        self.source_separator = SourceSeparator(self.device)

        # Compatibility layer to solve dimension mismatch issues - initialized by loading logic
        self.compat_mode = False
        self.dimension_adapter = None

        # Load saved model if provided
        if model_path is not None:
            load_status = self.persistence_manager.load_model(model_path)
            if load_status['success']:
                loaded_config = load_status.get('config', {})
                self.chinese_instruments = loaded_config.get('chinese_instruments', self.chinese_instruments)
                self.compat_mode = load_status.get('compat_mode', False)
                if self.compat_mode:
                    # Create a dimension adapter - for handling conversions between 3D and 4D tensors
                    self.dimension_adapter = DimensionAdapter().to(self.device)
                    logger.info("DimensionAdapter created due to compat_mode.") 
            else:
                logger.error(f"Model loading failed for path: {model_path}. Initializing with default configuration.")

        # Note: self._demucs_model is no longer part of InstrumentTimbreModel directly.
        # It's managed within self.source_separator.

        # Initialize encoder dictionary
        self.encoders = {
            "default": self.encoder,
            "chinese": self.encoder
            if chinese_instruments
            else ChineseInstrumentTimbreEncoder().to(self.device),
        }

    def load_model(self, model_path):
        """
        Load a saved model

        Args:
            model_path: Path to the saved model
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            try:
                # Attempt to load normally
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.decoder.load_state_dict(checkpoint["decoder"])
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                # If normal loading fails, enable compatibility mode
                logger.error(f"Error loading model from {model_path}: {e}")
                logger.info("Enabling compatibility mode for older model format")
                self.compat_mode = True

                # Create a dimension adapter - for handling conversions between 3D and 4D tensors
                self.dimension_adapter = DimensionAdapter().to(self.device)

                # Attempt to load partial weights
                self._load_partial_weights(checkpoint)

            # Load configuration if exists
            if "config" in checkpoint:
                config = checkpoint["config"]
                self.chinese_instruments = config.get(
                    "chinese_instruments", self.chinese_instruments
                )
    def separate_audio_sources(self, audio_file, output_dir=None):
        """
        Separate audio into different instrument sources using Demucs.
        This method is now a wrapper around SourceSeparator.separate_audio_sources.
        """
        return self.source_separator.separate_audio_sources(audio_file, output_dir)

    def extract_timbre(
        self,
        audio_path,
        segment_duration=3.0,
        hop_length=1.5,
        feature_level=2,
        return_all_segments=False,
        normalize=True,
        instrument_type=None,
    ):
        """
        Extract timbre features from audio file

        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            hop_length: Hop length between segments in seconds
            feature_level: Level of feature abstraction (1-3)
            return_all_segments: If True, return features for all segments
            normalize: If True, normalize features
            instrument_type: Specific instrument type to optimize extraction for

        Returns:
            Numpy array of timbre features
        """
        try:
            # Support direct input of audio data
            if isinstance(audio_path, torch.Tensor) or isinstance(
                audio_path, np.ndarray
            ):
                if isinstance(audio_path, np.ndarray):
                    audio_data = torch.from_numpy(audio_path).float()
                else:
                    audio_data = audio_path

                # If mono, add channel dimension
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)

                # Get sample rate - use default value
                sample_rate = 44100
            else:
                # Load audio from file
                audio_data, sample_rate = torchaudio.load(audio_path)

                # If stereo, convert to mono
                if audio_data.size(0) > 1:
                    audio_data = torch.mean(audio_data, dim=0, keepdim=True)

            # Divide audio into segments
            segment_length = int(segment_duration * sample_rate)
            hop_size = int(hop_length * sample_rate)
            segments = []

            # If audio is too short, pad with zeros
            if audio_data.size(1) < segment_length:
                padded = torch.zeros(
                    (audio_data.size(0), segment_length), device=audio_data.device
                )
                padded[:, : audio_data.size(1)] = audio_data
                segments.append(padded)
            else:
                # Split audio into overlapping segments
                for start in range(
                    0, audio_data.size(1) - segment_length + 1, hop_size
                ):
                    segments.append(audio_data[:, start : start + segment_length])

            # Extract features for each segment
            all_features = []
            for segment in segments:
                # Use Mel spectrum, chroma features, and MFCC as input features
                # Ensure segment is on the correct device for feature_processor methods
                segment_device = segment.device if isinstance(segment, torch.Tensor) else self.device
                
                mel_spec = self.feature_processor.to_mel_spectrogram(segment.to(self.device), sample_rate)
                chroma = self.feature_processor.to_chroma(segment.to(self.device), sample_rate) # to_chroma handles device internally for output
                mfcc = self.feature_processor.to_mfcc(segment.to(self.device), sample_rate)

                # Concatenate all features together [C, F, T]
                combined_features = torch.cat([mel_spec, chroma, mfcc], dim=0)

                # Check feature dimensions
                if (
                    torch.isnan(combined_features).any()
                    or torch.isinf(combined_features).any()
                ):
                    print(
                        f"Warning: NaN or Inf values in features. Using fallback method."
                    )
                    # Use simple features as a fallback
                    # Ensure segment is on the correct device for these direct transforms too
                    segment_on_device = segment.to(self.device)
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128
                    )(segment_on_device)
                    mel_spec = torch.log(mel_spec + 1e-9)
                    chroma = torch.zeros(
                        (12, mel_spec.shape[-1]), device=self.device # Use self.device for consistency
                    )
                    mfcc = torch.zeros((20, mel_spec.shape[-1]), device=self.device) # Use self.device
                    combined_features = torch.cat([mel_spec, chroma, mfcc], dim=0)

                try:
                    # Extract model features - pass features directly to the encoder
                    features = None
                    if instrument_type == "erhu" or "erhu" in str(audio_path).lower():
                        # Use Chinese traditional instrument encoder
                        try:
                            # First, try using the specialized encoder
                            encoder = self.encoders.get(
                                "chinese", self.encoders["default"]
                            )
                            features = encoder(combined_features)
                        except Exception as e:
                            print(
                                f"Chinese encoder failed: {e}. Trying default encoder."
                            )
                            # If it fails, try using the default encoder
                            encoder = self.encoders["default"]
                            features = encoder(combined_features)
                    else:
                        # Use the default encoder
                        encoder = self.encoders.get(
                            instrument_type, self.encoders["default"]
                        )
                        features = encoder(combined_features)

                    # Convert to numpy
                    if isinstance(features, torch.Tensor):
                        features = features.detach().cpu().numpy()

                    all_features.append(features)

                except Exception as e:
                    # If model extraction fails, use a fallback method
                    print(
                        f"Error extracting features with model: {e}. Using fallback method."
                    )

                    # Use traditional features as a fallback
                    # Calculate statistical features like mean and standard deviation
                    mean_features = combined_features.mean(dim=-1).cpu().numpy()
                    std_features = combined_features.std(dim=-1).cpu().numpy()
                    max_features = combined_features.max(dim=-1)[0].cpu().numpy()

                    # Combine statistical features
                    statistical_features = np.concatenate(
                        [mean_features, std_features, max_features]
                    )

                    # Adjust if a specific feature dimension is required
                    target_dim = 128  # Target dimension
                    if len(statistical_features) > target_dim:
                        # Dimensionality reduction
                        statistical_features = statistical_features[:target_dim]
                    elif len(statistical_features) < target_dim:
                        # Padding
                        pad_size = target_dim - len(statistical_features)
                        statistical_features = np.pad(
                            statistical_features, (0, pad_size), "constant"
                        )

                    all_features.append(statistical_features)

            # Combine features from all segments
            if return_all_segments:
                features = np.array(all_features)
            else:
                # Calculate the mean feature vector
                features = np.mean(all_features, axis=0)

            # Normalization
            if normalize and features.size > 0:
                features_mean = (
                    np.mean(features, axis=0)
                    if features.ndim > 1
                    else np.mean(features)
                )
                features_std = (
                    np.std(features, axis=0) if features.ndim > 1 else np.std(features)
                )
                # Avoid division by zero
                features_std = np.where(features_std < 1e-6, 1.0, features_std)
                features = (features - features_mean) / features_std

            return features

        except Exception as e:
            logger.error(f"Error in extract_timbre: {e}", exc_info=True) # Log with stack trace
            logger.info("All primary and statistical fallback feature extraction attempts failed within extract_timbre.")
            # As per current design, _extract_fallback_features is not directly called as a fallback
            # by the main extract_timbre method's internal loop.
            # If it were, the logic would be here to call it.
            # For now, returning None as the ultimate fallback.
            return None

    def _extract_fallback_features(self, audio, sr, audio_file, output_dir):
        """Fallback feature extraction method, used when model extraction fails"""
        logger.info("Using fallback feature extraction method")

        try:
            # Determine instrument category
            instrument_category = "bowed_string"  # Default to bowed string (Erhu)
            file_lower = audio_file.lower()
            if "erhu" in file_lower or "二胡" in file_lower: # "二胡" means Erhu
                instrument_category = "bowed_string"  # Bowed string
            elif "pipa" in file_lower or "琵琶" in file_lower: # "琵琶" means Pipa
                instrument_category = "plucked_string"  # Plucked string
            elif "dizi" in file_lower or "笛子" in file_lower: # "笛子" means Dizi (flute)
                instrument_category = "wind"  # Wind instrument

            # Use direct feature extraction method
            # Ensure audio is a NumPy array for extract_chinese_instrument_features if it expects that
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
                if audio_np.ndim > 1 and audio_np.shape[0] == 1: # Mono tensor [1, N]
                    audio_np = audio_np.squeeze(0)
            else:
                audio_np = audio # Assuming it's already a numpy array or compatible

            features = extract_chinese_instrument_features(
                audio_np, sr, instrument_category=instrument_category
            )

            # Convert features to an embedding vector
            spectral_features = np.concatenate(
                [
                    features["spectral_centroid"].reshape(-1)[:32],  # Take first 32 spectral centroid features
                    features["spectral_contrast"].reshape(-1)[:32],  # Take first 32 spectral contrast features
                    np.mean(features["harmonic_component"], axis=0)[:32],  # Take first 32 harmonic component features
                    features["pitch_delta_stats"],  # Pitch variation statistics
                ]
            )

            # Pad or truncate to 128 dimensions
            embedding_size = 128
            if len(spectral_features) < embedding_size:
                embedding = np.pad(
                    spectral_features, (0, embedding_size - len(spectral_features))
                )
            else:
                embedding = spectral_features[:embedding_size]

            # Create result
            result = {"embedding": embedding, "features": features}

            # Save feature file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                feature_file = os.path.join(
                    output_dir, f"{base_name}_timbre_embedding.npy"
                )
                np.save(feature_file, embedding)
                result["feature_file"] = feature_file
                logger.info(f"Saved fallback timbre features to {feature_file}")

            return result

        except Exception as e:
            logger.error(f"Fallback feature extraction failed: {e}", exc_info=True)
            return None

    def apply_timbre(
        self, target_file, timbre_features, output_dir=None, intensity=0.8
    ):
        """
        Apply extracted timbre to a target audio file

        Args:
            target_file: Path to target audio file
            timbre_features: Path to timbre features file or feature dictionary
            output_dir: Directory to save output audio
            intensity: Timbre application intensity (0.0-1.0)

        Returns:
            Path to output audio file
        """
        # Use target file's directory if output_dir not specified
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(target_file), "transformed")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(target_file))[0]
        timbre_name = (
            os.path.splitext(os.path.basename(timbre_features))[0]
            if isinstance(timbre_features, str)
            else "applied_timbre"
        )
        output_file = os.path.join(output_dir, f"{base_name}_with_{timbre_name}.wav")

        try:
            # Load target audio
            logger.info(f"Loading target audio file: {target_file}")
            target_audio, sr = load_audio(target_file, sr=22050, mono=True)

            if target_audio is None:
                logger.error(f"Failed to load target audio file: {target_file}")
                return None

            # Load timbre features
            if isinstance(timbre_features, str):
                logger.info(f"Loading timbre features from: {timbre_features}")
                timbre_data = np.load(timbre_features)
                timbre_vector = timbre_data["timbre_vector"]
            elif (
                isinstance(timbre_features, dict) and "timbre_vector" in timbre_features
            ):
                timbre_vector = timbre_features["timbre_vector"]
            else:
                logger.error("Invalid timbre features")
                return None

            # Extract target audio features
            logger.info("Extracting features from target audio")
            target_mel = extract_features(target_audio, sr, feature_type="mel")
            target_mel_tensor = (
                torch.from_numpy(target_mel).float().unsqueeze(0).to(self.device)
            )

            # Convert timbre vector to tensor
            timbre_tensor = torch.from_numpy(timbre_vector).float().to(self.device)

            # Run through decoder to generate transformed spectrogram
            logger.info("Applying timbre transformation")
            self.decoder.eval()
            with torch.no_grad():
                transformed_mel = self.decoder(timbre_tensor)

                # Apply with specified intensity
                transformed_mel = (
                    1 - intensity
                ) * target_mel_tensor + intensity * transformed_mel

            # Convert back to audio (simplified - in a real system you'd use a spectrogram inversion method)
            # For demonstration purposes, we'll use the Griffin-Lim algorithm from librosa
            import librosa

            # Convert mel spectrogram back to magnitude spectrogram
            transformed_mel_np = transformed_mel.cpu().numpy()[
                0
            ]  # Remove batch dimension
            S = librosa.feature.inverse.mel_to_stft(transformed_mel_np, sr=sr)

            # Griffin-Lim for phase reconstruction
            logger.info("Converting spectrogram back to audio")
            transformed_audio = librosa.griffinlim(S)

            # Save the transformed audio
            logger.info(f"Saving transformed audio to: {output_file}")
            save_audio(transformed_audio, output_file, sr)

            return output_file

        except Exception as e:
            logger.error(f"Error applying timbre: {e}")
            return None

    def train(self, dataloader, epochs=10, learning_rate=0.001, max_batches=None):
        """Train the model"""
        device = self.device
        self.encoder.train()
        self.decoder.train()

        # Use simple MSE loss for initial training stability
        criterion = nn.MSELoss()

        # Lower initial learning rate for more stable convergence

        # Reduce learning rate to improve stability
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
            * 0.001,  # Reduce learning rate significantly for stability
        )

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5, verbose=True
        )

        # Enable TensorBoard logging
        from torch.utils.tensorboard import SummaryWriter
        import time

        self.writer = SummaryWriter(f"logs/timbre_model_{int(time.time())}")
        logger.info(f"TensorBoard logging enabled at {self.writer.log_dir}")

        # AMP setup
        use_amp = torch.cuda.is_available()
        if use_amp:
            scaler = GradScaler()
            logger.info("AMP enabled for training as CUDA is available.")
        else:
            logger.info("AMP not enabled for training (CUDA not available or use_amp=False).")


        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (mel_spec, _) in progress_bar:
                # Stop after max_batches if specified
                if max_batches and i >= max_batches:
                    break

                # Move data to device
                mel_spec = mel_spec.to(device)

                # Normalize mel spectrogram with robust scaling to prevent numerical instability
                # First ensure no NaN or Inf values
                mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=1.0, neginf=0.0)

                # Apply robust min-max scaling with small epsilon
                eps = 1e-5
                mel_min = mel_spec.min()
                mel_max = mel_spec.max()
                if mel_max > mel_min:
                    mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + eps)
                else:
                    # If constant values, just zero out to avoid NaN
                    mel_spec = torch.zeros_like(mel_spec)
                
                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    # Forward pass through encoder and decoder
                    # The mel_spec shape should be [batch_size, 1, height, width]
                    # Make sure mel_spec has the right shape before passing to encoder
                    if mel_spec.dim() != 4:
                        raise ValueError(
                            f"Expected 4D tensor [batch_size, channels, height, width], got shape: {mel_spec.shape}"
                        )

                    # Ensure mel_spec has the correct channel dimension
                    if mel_spec.shape[1] != 1:
                        print(
                            f"WARNING: Expected 1 channel, got {mel_spec.shape[1]}. Reshaping..."
                        )
                        # Take only the first item from each batch and reshape
                        mel_spec = mel_spec[:, 0:1, :, :]
                        print(f"New shape: {mel_spec.shape}")

                    timbre_vector = self.encoder(mel_spec)
                    reconstructed = self.decoder(timbre_vector)

                    # Compute loss
                    loss = criterion(reconstructed, mel_spec)

                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient Clipping (after unscaling if using AMP)
                if use_amp:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=0.5,  # Lower max_norm for more stability
                )

                # Check for NaN in gradients and skip step if found
                skip_step = False
                for param in list(self.encoder.parameters()) + list(
                    self.decoder.parameters()
                ):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        skip_step = True
                        logger.warning("NaN detected in gradients, skipping step")
                        break
                
                # Optimizer step
                if not skip_step:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                # Log statistics
                total_loss += loss.item()
                total_batches += 1

                # Update progress bar
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
                progress_bar.set_postfix(loss=loss.item())

                # Log to TensorBoard
                step = epoch * len(dataloader) + i
                self.writer.add_scalar("Loss/train", loss.item(), step)

                # Log examples periodically
                if i % 10 == 0:
                    # Log example spectrograms
                    self.writer.add_image(
                        "Spectrograms/original",
                        mel_spec[0].detach().cpu().numpy(),
                        step,
                        dataformats="CHW",
                    )
                    self.writer.add_image(
                        "Spectrograms/reconstructed",
                        reconstructed[0].detach().cpu().numpy(),
                        step,
                        dataformats="CHW",
                    )

            # Epoch statistics
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)

            # Update learning rate scheduler
            scheduler.step(avg_loss)
        
        # Save model after training - using the persistence manager
        # Assuming args.model_path is available in this scope or passed appropriately
        # For now, let's assume self.last_model_save_path is set during train command in app.py
        # Or, train() should accept save_path as an argument.
        # For this refactor, I'll assume that the calling context (e.g. app.py)
        # will call persistence_manager.save_model after train() completes.
        # If train itself must save, it needs the path and config.
        # The original code called self.save_model(args.model_path) from app.py after model.train(...)
        # This means InstrumentTimbreModel.train() does not need to save the model itself.
        # However, the original InstrumentTimbreModel.train() had a writer.close()
        # which implies it considered itself the end of the training process.

        # Close TensorBoard writer
        self.writer.close()


# Add a dimension adapter class to handle conversions between 3D and 4D tensors
class DimensionAdapter(nn.Module):
    """Adapter class to handle dimension differences between old and new model formats"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Adapt dimensions as needed"""
        if x.dim() == 3 and x.size(0) == 1:  # [1, seq_len, features]
            return x.squeeze(0)  # Convert to [seq_len, features]
        elif x.dim() == 2:  # [seq_len, features]
            return x.unsqueeze(0)  # Convert to [1, seq_len, features]
        return x  # Pass through unchanged
