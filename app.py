#!/usr/bin/env python
"""
Instrument Timbre Analysis and Manipulation - Main Application Entry

This module provides command-line interface for training, feature extraction, model export, and timbre application.
"""
import os
import argparse
import logging
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from InstrumentTimbre.models.model import InstrumentTimbreModel
from InstrumentTimbre.utils.export import ModelExporter

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataloader(
    data_dir, batch_size=32, chinese_instruments=False, feature_type="multi", **kwargs
):
    """
    Prepare data loader based on configuration
    """
    if chinese_instruments:
        from InstrumentTimbre.utils.data import prepare_chinese_instrument_dataloader

        return prepare_chinese_instrument_dataloader(
            data_dir, batch_size=batch_size, feature_type=feature_type, **kwargs
        )
    else:
        from InstrumentTimbre.utils.data import prepare_dataloader

        return prepare_dataloader(data_dir, batch_size=batch_size, **kwargs)


def main():
    """
    Main command-line entry. Parses arguments and calls corresponding functions based on sub-commands.
    Supports: training, feature extraction, model export, timbre application, etc.
    """
    parser = argparse.ArgumentParser(
        description="Instrument Timbre Analysis and Manipulation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train timbre model")
    train_parser.add_argument(
        "--data-dir", "--dataset-path", default="../wav", help="Training data directory"
    )
    train_parser.add_argument("--model-path", required=True, help="Model save path")
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    train_parser.add_argument(
        "--use-wav-files",
        action="store_true",
        help="Use WAV files directly as training data",
    )
    train_parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    train_parser.add_argument(
        "--chinese-instruments",
        action="store_true",
        help="Optimize for Chinese traditional instruments",
    )
    train_parser.add_argument(
        "--feature-type",
        choices=["mel", "constant-q", "multi"],
        default="multi",
        help="Feature type (Chinese instruments only)",
    )
    train_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained audio models"
    )
    train_parser.add_argument(
        "--cache-features", action="store_true", help="Enable feature caching"
    )
    train_parser.add_argument(
        "--export-onnx", action="store_true", help="Export trained model to ONNX format"
    )

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract timbre features")
    extract_parser.add_argument("--model-path", help="Model path")
    extract_parser.add_argument(
        "--input-file", "--source-file", required=True, help="Input audio file"
    )
    extract_parser.add_argument(
        "--output-dir", "--output-file", help="Output directory"
    )
    extract_parser.add_argument(
        "--chinese-instruments",
        action="store_true",
        help="Use Chinese instrument models",
    )
    extract_parser.add_argument(
        "--cache-features", action="store_true", help="Enable feature caching"
    )

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply timbre features")
    apply_parser.add_argument("--model-path", help="Model path")
    apply_parser.add_argument("--target-file", required=True, help="Target audio file")
    apply_parser.add_argument(
        "--timbre-file", required=True, help="Timbre feature file"
    )
    apply_parser.add_argument("--output-dir", help="Output directory")
    apply_parser.add_argument(
        "--intensity",
        type=float,
        default=0.8,
        help="Timbre application intensity (0.0-1.0)",
    )

    # Separate command for source separation
    separate_parser = subparsers.add_parser(
        "separate", help="Separate audio sources using Demucs"
    )
    separate_parser.add_argument("--input-file", required=True, help="Input audio file")
    separate_parser.add_argument("--output-dir", help="Output directory")

    # Replace instrument command
    replace_parser = subparsers.add_parser(
        "replace", help="Replace instrument in a mixed recording"
    )
    replace_parser.add_argument("--model-path", help="Model path")
    replace_parser.add_argument(
        "--input-file", required=True, help="Input mixed audio file"
    )
    replace_parser.add_argument(
        "--target-instrument-file",
        required=True,
        help="Audio file with target instrument timbre",
    )
    replace_parser.add_argument(
        "--source-type",
        default="other",
        choices=["vocals", "drums", "bass", "other"],
        help="Source type to replace",
    )
    replace_parser.add_argument("--output-dir", help="Output directory")
    replace_parser.add_argument(
        "--intensity",
        type=float,
        default=0.8,
        help="Timbre application intensity (0.0-1.0)",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export model to different formats"
    )
    export_parser.add_argument("--model-path", required=True, help="Source model path")
    export_parser.add_argument(
        "--output-path", required=True, help="Output path for exported model"
    )
    export_parser.add_argument(
        "--format",
        required=True,
        choices=["onnx", "torchscript", "quantized"],
        help="Export format",
    )
    export_parser.add_argument(
        "--input-shape",
        default="1,1,128,128",
        help="Input shape for model (comma-separated)",
    )

    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage feature cache")
    cache_parser.add_argument(
        "--action", required=True, choices=["clear", "stats"], help="Cache action"
    )
    cache_parser.add_argument("--cache-dir", help="Custom cache directory")

    args = parser.parse_args()

    if args.command == "train":
        # Create model
        model = InstrumentTimbreModel(
            use_pretrained=args.pretrained,
            chinese_instruments=args.chinese_instruments,
            feature_caching=args.cache_features,
        )

        # Prepare data loader
        dataloader = get_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            chinese_instruments=args.chinese_instruments,
            feature_type=args.feature_type,
            use_wav_files=args.use_wav_files,
            augment=args.augment,
            debug=args.debug,
        )

        # Train model
        model.train(
            dataloader,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_batches=100 if args.debug else None,
        )

        # Save model
        model.save_model(args.model_path)
        logger.info(f"Model saved to {args.model_path}")

        # Export to ONNX if requested
        if args.export_onnx:
            onnx_path = os.path.splitext(args.model_path)[0] + ".onnx"
            if ModelExporter.to_onnx(model.encoder, onnx_path):
                logger.info(f"Model exported to ONNX format: {onnx_path}")

    elif args.command == "extract":
        # Load model
        model = InstrumentTimbreModel(
            model_path=args.model_path,
            chinese_instruments=args.chinese_instruments,
            feature_caching=args.cache_features,
        )

        # Extract timbre features
        # The model.extract_timbre method returns features (numpy array) or None.
        # It does not handle saving features itself.
        instrument_type_arg = "chinese" if args.chinese_instruments else None
        
        logger.info(f"Extracting timbre features from: {args.input_file}")
        features = model.extract_timbre(
            audio_path=args.input_file,
            instrument_type=instrument_type_arg
        )

        if features is not None:
            logger.info(f"Successfully extracted timbre features from {args.input_file}.")
            if args.output_dir:
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
                    output_feature_file = os.path.join(args.output_dir, f"{base_name}_timbre_features.npy")
                    
                    # Need numpy to save features
                    import numpy as np
                    np.save(output_feature_file, features)
                    logger.info(f"Timbre features saved to: {output_feature_file}")
                except Exception as e:
                    logger.error(f"Failed to save timbre features to {args.output_dir}: {e}", exc_info=True)
            else:
                logger.info("--output-dir not specified, features were extracted but not saved.")
        else:
            logger.error(f"Failed to extract timbre features from {args.input_file} after all attempts. No features were saved.")

    elif args.command == "apply":
        # Load model
        model = InstrumentTimbreModel(model_path=args.model_path)

        # Apply timbre
        output_file = model.apply_timbre(
            args.target_file,
            args.timbre_file,
            args.output_dir,
            intensity=args.intensity,
        )
        if output_file:
            logger.info(f"Transformed audio saved to: {output_file}")
        else:
            logger.error("Failed to apply timbre transformation")

    elif args.command == "separate":
        # Create model for source separation
        model = InstrumentTimbreModel()

        # Separate audio sources
        result = model.separate_audio_sources(args.input_file, args.output_dir)
        if result:
            logger.info(f"Audio sources separated and saved to: {args.output_dir}")
            for source, path in result["sources"].items():
                logger.info(f"  - {source}: {path}")
        else:
            logger.error("Failed to separate audio sources")

    elif args.command == "replace":
        # This command would combine source separation and timbre transfer
        # Implementation would go here
        logger.info("Instrument replacement functionality not implemented yet")
        # In a real implementation, this would:
        # 1. Separate the mixed audio into individual tracks
        # 2. Extract timbre from target instrument file
        # 3. Apply timbre to the selected instrument track
        # 4. Recombine all tracks

    elif args.command == "export":
        # Parse input shape
        input_shape = tuple(map(int, args.input_shape.split(",")))

        # Load model
        model = InstrumentTimbreModel(model_path=args.model_path)

        # Export model based on format
        if args.format == "onnx":
            success = ModelExporter.to_onnx(
                model.encoder, args.output_path, input_shape=input_shape
            )
        elif args.format == "torchscript":
            success = ModelExporter.to_torchscript(model.encoder, args.output_path)
        elif args.format == "quantized":
            success = ModelExporter.to_quantized(model.encoder, args.output_path)
        else:
            logger.error(f"Unsupported export format: {args.format}")
            success = False

        if success:
            logger.info(f"Model exported to {args.format} format: {args.output_path}")
        else:
            logger.error(f"Failed to export model to {args.format} format")

    elif args.command == "cache":
        # Create a feature cache instance
        from InstrumentTimbre.utils.cache import FeatureCache

        cache = FeatureCache(args.cache_dir)

        if args.action == "clear":
            # Clear cache
            if cache.clear():
                logger.info("Feature cache cleared successfully")
            else:
                logger.error("Failed to clear feature cache")

        elif args.action == "stats":
            # Display cache statistics
            stats = cache.stats()
            logger.info("Feature cache statistics:")
            logger.info(f"  - Cache directory: {stats['cache_dir']}")
            logger.info(f"  - Total entries: {stats['entries']}")
            logger.info(f"  - Valid files: {stats['valid_files']}")
            logger.info(f"  - Total size: {stats['total_size_mb']:.2f} MB")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
