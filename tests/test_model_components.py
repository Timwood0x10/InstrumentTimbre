import pytest
import torch
import os
import sys

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from InstrumentTimbre.models.model import AudioFeatureProcessor, InstrumentTimbreModel
# Assuming FeatureCache might be needed for some tests later, or if model instantiation requires it
from InstrumentTimbre.utils.cache import FeatureCache


def test_audio_feature_processor_initialization_and_mel():
    """
    Tests the initialization of AudioFeatureProcessor and its to_mel_spectrogram method.
    """
    # Instantiate AudioFeatureProcessor (default device is cpu, no caching needed for this test)
    # Default sample_rate=44100, n_mels=128
    feature_processor = AudioFeatureProcessor(feature_cache=None)

    # Create a dummy mono audio tensor
    # Using a sample rate that AudioFeatureProcessor's transforms expect by default (44100 Hz)
    # Duration of approx 1 second
    sample_rate = 44100
    dummy_audio = torch.randn(1, sample_rate) 

    # Call to_mel_spectrogram
    # The method expects sample_rate of the audio if different from its internal default.
    # If audio is already at target SR, it's fine.
    mel_output = feature_processor.to_mel_spectrogram(dummy_audio, sample_rate=sample_rate)

    assert isinstance(mel_output, torch.Tensor), "Output should be a torch.Tensor"

    # Expected shape [n_mels, time_frames]
    # n_mels is 128 by default in AudioFeatureProcessor
    # Number of time_frames depends on n_fft, hop_length, and input length
    assert mel_output.ndim == 2, f"Mel spectrogram should have 2 dimensions, got {mel_output.ndim}"
    assert mel_output.shape[0] == feature_processor.n_mels, \
        f"Number of mel bands should be {feature_processor.n_mels}, got {mel_output.shape[0]}"
    assert mel_output.shape[1] > 0, "Number of time frames should be greater than 0"

    # Test with a different sample rate to ensure resampling works
    sample_rate_input = 16000
    dummy_audio_16k = torch.randn(1, sample_rate_input)
    mel_output_16k = feature_processor.to_mel_spectrogram(dummy_audio_16k, sample_rate=sample_rate_input)
    
    assert isinstance(mel_output_16k, torch.Tensor), "Output from 16k audio should be a torch.Tensor"
    assert mel_output_16k.ndim == 2, f"Mel spectrogram from 16k audio should have 2 dimensions, got {mel_output_16k.ndim}"
    assert mel_output_16k.shape[0] == feature_processor.n_mels, \
        f"Number of mel bands from 16k audio should be {feature_processor.n_mels}, got {mel_output_16k.shape[0]}"
    assert mel_output_16k.shape[1] > 0, "Number of time frames from 16k audio should be greater than 0"


def test_extract_timbre_handles_invalid_file():
    """
    Tests that InstrumentTimbreModel.extract_timbre returns None for a non-existent audio file.
    """
    # Instantiate InstrumentTimbreModel
    # feature_caching=False to avoid cache creation/interaction for this test
    # model_path=None as we are not testing a pre-trained model's specific output here
    model = InstrumentTimbreModel(model_path=None, feature_caching=False)

    # Call extract_timbre with a non-existent file
    # torchaudio.load inside extract_timbre will raise an error, which should be caught.
    features = model.extract_timbre("non_existent_audio_file.wav")

    assert features is None, "extract_timbre should return None for non-existent files"

# Example of a simple test for Chroma and MFCC, can be expanded
def test_audio_feature_processor_chroma_mfcc():
    """
    Basic tests for to_chroma and to_mfcc methods of AudioFeatureProcessor.
    """
    feature_processor = AudioFeatureProcessor(feature_cache=None)
    sample_rate = 44100
    dummy_audio = torch.randn(1, sample_rate)

    # Test to_chroma
    chroma_output = feature_processor.to_chroma(dummy_audio, sample_rate=sample_rate)
    assert isinstance(chroma_output, torch.Tensor)
    assert chroma_output.ndim == 2
    assert chroma_output.shape[0] == feature_processor.n_chroma # Default 12
    assert chroma_output.shape[1] > 0

    # Test to_mfcc
    mfcc_output = feature_processor.to_mfcc(dummy_audio, sample_rate=sample_rate)
    assert isinstance(mfcc_output, torch.Tensor)
    assert mfcc_output.ndim == 2
    assert mfcc_output.shape[0] == feature_processor.n_mfcc # Default 20
    assert mfcc_output.shape[1] > 0

# It's good practice to test caching if it's a critical feature
# For this, we might need a temporary directory and actual file operations.
# This is a more advanced test.

# @pytest.fixture
# def temp_cache_dir(tmp_path):
#     """Create a temporary cache directory for testing FeatureCache."""
#     cache_dir = tmp_path / "feature_cache"
#     cache_dir.mkdir()
#     return cache_dir

# def test_audio_feature_processor_chroma_with_caching(temp_cache_dir):
#     """
#     Tests to_chroma with caching enabled.
#     This test requires a valid audio file that can be loaded by `load_audio`.
#     For simplicity, this is a placeholder structure, as it needs a sample audio file.
#     """
#     # Create a dummy wav file in the temp_cache_dir (or a general temp resources dir for tests)
#     import soundfile as sf
#     dummy_audio_path = temp_cache_dir / "test.wav"
#     sample_rate = 44100
#     data = torch.randn(1, sample_rate).numpy().T # soundfile expects [frames, channels] or [frames]
#     sf.write(dummy_audio_path, data, sample_rate)

#     feature_cache = FeatureCache(cache_dir=str(temp_cache_dir))
#     feature_processor = AudioFeatureProcessor(feature_cache=feature_cache)

#     # First call - should compute and cache
#     chroma_output1 = feature_processor.to_chroma(str(dummy_audio_path), sample_rate=sample_rate)
#     assert isinstance(chroma_output1, torch.Tensor)
#     assert (temp_cache_dir / "chroma_cache" / "test.wav.pt").exists(), "Chroma cache file should be created"

#     # Second call - should load from cache
#     # To truly test caching, one might mock `chroma_transform` to see if it's NOT called again.
#     # Or, check modification times / content if possible.
#     chroma_output2 = feature_processor.to_chroma(str(dummy_audio_path), sample_rate=sample_rate)
#     assert torch.equal(chroma_output1, chroma_output2), "Cached output should match original"

#     # Clean up cache explicitly if needed, though tmp_path handles removal
#     feature_cache.clear_chroma()
#     assert not (temp_cache_dir / "chroma_cache" / "test.wav.pt").exists(), "Chroma cache file should be deleted"

if __name__ == "__main__":
    # This allows running pytest directly on this file if needed
    # For example: python tests/test_model_components.py
    # However, usually, you'd run `pytest` from the project root.
    pytest.main([__file__])


# --- Tests for ModelPersistenceManager ---
from unittest.mock import patch, MagicMock
from InstrumentTimbre.models.model import ModelPersistenceManager

@pytest.fixture
def mock_models_and_device():
    """Pytest fixture to create mock encoder, decoder, and device."""
    mock_encoder = MagicMock(spec=torch.nn.Module)
    mock_encoder.state_dict = MagicMock(return_value={"encoder_param": torch.tensor([1.0])})
    
    mock_decoder = MagicMock(spec=torch.nn.Module)
    mock_decoder.state_dict = MagicMock(return_value={"decoder_param": torch.tensor([2.0])})
    
    mock_device = MagicMock(spec=torch.device)
    return mock_encoder, mock_decoder, mock_device

def test_mpm_initialization(mock_models_and_device):
    """Test ModelPersistenceManager initialization."""
    mock_encoder, mock_decoder, mock_device = mock_models_and_device
    mpm = ModelPersistenceManager(mock_encoder, mock_decoder, mock_device)
    assert mpm.encoder == mock_encoder
    assert mpm.decoder == mock_decoder
    assert mpm.device == mock_device

@patch('torch.save')
def test_mpm_save_model(mock_torch_save, mock_models_and_device):
    """Test ModelPersistenceManager save_model method."""
    mock_encoder, mock_decoder, mock_device = mock_models_and_device
    mpm = ModelPersistenceManager(mock_encoder, mock_decoder, mock_device)
    
    test_path = "dummy_save_path.pt"
    test_config = {"chinese_instruments": True, "version": "1.0"}
    
    result = mpm.save_model(test_path, test_config)
    
    assert result is True
    mock_torch_save.assert_called_once()
    args, _ = mock_torch_save.call_args
    saved_checkpoint = args[0]
    saved_path = args[1]
    
    assert saved_path == test_path
    assert "encoder" in saved_checkpoint
    assert "decoder" in saved_checkpoint
    assert "config" in saved_checkpoint
    assert saved_checkpoint["config"] == test_config
    mock_encoder.state_dict.assert_called_once()
    mock_decoder.state_dict.assert_called_once()

@patch.object(ModelPersistenceManager, '_load_partial_weights_in_manager') # Mocking this to simplify test
@patch('torch.load')
def test_mpm_load_model_success(mock_torch_load, mock_load_partial, mock_models_and_device):
    """Test ModelPersistenceManager load_model success and config loading."""
    mock_encoder, mock_decoder, mock_device = mock_models_and_device
    
    # Mock what torch.load would return
    mock_checkpoint_data = {
        "encoder": {"encoder_param": torch.tensor([1.0])},
        "decoder": {"decoder_param": torch.tensor([2.0])},
        "config": {"chinese_instruments": False, "loaded_version": "0.9"}
    }
    mock_torch_load.return_value = mock_checkpoint_data
    
    mpm = ModelPersistenceManager(mock_encoder, mock_decoder, mock_device)
    status = mpm.load_model("dummy_load_path.pt")
    
    assert status['success'] is True
    assert status['config'] == mock_checkpoint_data['config']
    assert status['compat_mode'] is False # Assuming normal load path
    
    mock_torch_load.assert_called_once_with("dummy_load_path.pt", map_location=mock_device)
    mock_encoder.load_state_dict.assert_called_once_with(mock_checkpoint_data["encoder"])
    mock_decoder.load_state_dict.assert_called_once_with(mock_checkpoint_data["decoder"])
    mock_load_partial.assert_not_called() # Should not be called if direct load_state_dict succeeds


@patch('torch.load')
def test_mpm_load_model_compat_mode(mock_torch_load, mock_models_and_device):
    """Test ModelPersistenceManager load_model with compatibility mode triggered."""
    mock_encoder, mock_decoder, mock_device = mock_models_and_device
    
    # Simulate load_state_dict failing to trigger compat mode
    mock_encoder.load_state_dict.side_effect = RuntimeError("Simulated state_dict load error")

    mock_checkpoint_data = {
        "encoder": {"some_old_param": torch.tensor([1.0])}, # Old format
        "decoder": {"some_other_param": torch.tensor([2.0])},
        "config": {"chinese_instruments": True}
    }
    mock_torch_load.return_value = mock_checkpoint_data
    
    mpm = ModelPersistenceManager(mock_encoder, mock_decoder, mock_device)
    
    # Patch _load_partial_weights_in_manager for this specific call to see if it's invoked
    with patch.object(mpm, '_load_partial_weights_in_manager') as mock_partial_loader:
        status = mpm.load_model("dummy_compat_path.pt")
    
    assert status['success'] is True
    assert status['compat_mode'] is True
    assert status['config'] == mock_checkpoint_data['config']
    mock_partial_loader.assert_called_once_with(mock_checkpoint_data)


# --- Tests for SourceSeparator ---
from InstrumentTimbre.models.model import SourceSeparator # Already imported but good for clarity

def test_ss_initialization():
    """Test SourceSeparator initialization."""
    mock_device = MagicMock(spec=torch.device)
    ss = SourceSeparator(mock_device)
    assert ss.device == mock_device
    assert ss._demucs_model is None
    assert ss._apply_demucs is None


@patch('torchaudio.save')
@patch('torchaudio.load')
@patch.object(SourceSeparator, '_load_demucs_model')
def test_ss_separate_audio_sources_success(mock_load_demucs, mock_torchaudio_load, mock_torchaudio_save):
    """Test SourceSeparator separate_audio_sources successful execution."""
    mock_device = MagicMock(spec=torch.device)
    ss = SourceSeparator(mock_device)

    # Setup mocks for successful separation
    mock_load_demucs.return_value = True # Demucs model loaded successfully

    # Mock torchaudio.load
    dummy_audio_tensor = torch.randn(1, 88200) # Mono audio at 44.1kHz for 2s
    mock_torchaudio_load.return_value = (dummy_audio_tensor, 22050) # Simulate loading 22.05kHz audio

    # Mock Demucs model and its apply method (which is stored in _apply_demucs)
    ss._demucs_model = MagicMock()
    ss._demucs_model.sources = ['drums', 'bass', 'other', 'vocals'] # Expected source names
    # _apply_demucs should return a tensor of shape [num_sources, num_channels, num_samples]
    # For Demucs, output is stereo by default.
    mock_separated_sources_tensor = torch.randn(len(ss._demucs_model.sources), 2, 88200) 
    ss._apply_demucs = MagicMock(return_value=mock_separated_sources_tensor)
    
    # Mock os.path.splitext and os.path.basename for filename generation
    with patch('os.path.splitext', return_value=('dummy_audio', '.wav')), \
         patch('os.path.basename', return_value='dummy_audio.wav'):
        result = ss.separate_audio_sources("dummy_audio.wav", "output_test_dir")

    assert mock_load_demucs.called_once()
    mock_torchaudio_load.assert_called_once_with("dummy_audio.wav")
    ss._apply_demucs.assert_called_once() # Check if the separation was attempted
    
    # Check if torchaudio.save was called for each source
    assert mock_torchaudio_save.call_count == len(ss._demucs_model.sources)
    
    assert result is not None
    assert "sources" in result
    assert "source_names" in result
    assert len(result["sources"]) == len(ss._demucs_model.sources)
    for name in ss._demucs_model.sources:
        assert name in result["sources"]
        expected_path = os.path.join("output_test_dir", f"dummy_audio_{name}.wav")
        assert result["sources"][name] == expected_path


@patch('demucs.pretrained.get_model', side_effect=ImportError("Simulated Demucs import error"))
def test_ss_load_demucs_model_import_error(mock_get_model):
    """Test _load_demucs_model when demucs import fails."""
    mock_device = MagicMock(spec=torch.device)
    ss = SourceSeparator(mock_device)
    
    result = ss._load_demucs_model()
    
    assert result is False
    assert ss._demucs_model is None
    mock_get_model.assert_called_once() # Ensure it attempted to load


def test_ss_separate_audio_sources_load_demucs_fails():
    """Test separate_audio_sources when _load_demucs_model returns False."""
    mock_device = MagicMock(spec=torch.device)
    ss = SourceSeparator(mock_device)

    with patch.object(ss, '_load_demucs_model', return_value=False):
        result = ss.separate_audio_sources("dummy.wav", "out_dir")
        
    assert result is None

