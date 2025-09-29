"""Command-line interface for the DSPy optimization toolkit.

This module provides the CLI for compiling Claimify components using DSPy,
allowing users to optimize their own datasets and models.
"""

import argparse
import importlib.resources as resources
import shutil
import sys
from pathlib import Path

from .config import load_claimify_config, load_optimization_config


try:
    from .optimization.compile import (
        compile_component,
        DataValidationError,
        ModelConfigError,
        DSPyVersionError,
        OptimizationError,
    )
    from .optimization.data import print_schema_help
    from .optimization.generate import generate_dataset, GenerationError
except ImportError:

    def _missing_optimization_deps(*args, **kwargs):
        print(
            "❌ Error: DSPy optimization features are not available.", file=sys.stderr
        )
        print("💡 Install optimization dependencies with:", file=sys.stderr)
        print("   pip install 'aclarai-claimify[optimization]'", file=sys.stderr)
        sys.exit(1)

    compile_component = _missing_optimization_deps
    print_schema_help = _missing_optimization_deps
    generate_dataset = _missing_optimization_deps

    class DataValidationError(Exception):
        pass

    class ModelConfigError(Exception):
        pass

    class DSPyVersionError(Exception):
        pass

    class OptimizationError(Exception):
        pass

    class GenerationError(Exception):
        pass


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="aclarai-claimify",
        description="DSPy optimization toolkit for Claimify pipeline components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile selection component
  aclarai-claimify compile 
    --component selection 
    --trainset data/selection_train.jsonl 
    --student-model gpt-3.5-turbo 
    --teacher-model gpt-4o 
    --config optimization_configs/bootstrap_fewshot.yaml 
    --output-path artifacts/selection.json

  # Compile with different optimizer configuration
  aclarai-claimify compile 
    --component decomposition 
    --trainset data/decomposition_train.jsonl 
    --student-model gpt-3.5-turbo 
    --teacher-model gpt-4o 
    --config optimization_configs/mipro.yaml 
    --output-path artifacts/decomposition.json 
    --seed 123

  # Show schema help
  aclarai-claimify schema --component selection
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Create a local `settings` directory with default configurations",
        description="This command copies the default configuration files to your current "
        "directory, allowing you to easily customize them.",
    )
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing `settings` directory if it exists",
    )

    # Compile subcommand
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a component using DSPy optimization",
        description="Optimize a Claimify component using your training data and models",
    )

    # Required arguments
    compile_parser.add_argument(
        "--component",
        choices=["selection", "disambiguation", "decomposition"],
        required=True,
        help="Component to compile",
    )
    compile_parser.add_argument(
        "--trainset",
        type=Path,
        required=True,
        help="Path to training dataset (JSONL format)",
    )
    compile_parser.add_argument(
        "--student-model",
        required=True,
        help="Model for final program execution (e.g., gpt-3.5-turbo)",
    )
    compile_parser.add_argument(
        "--teacher-model",
        required=True,
        help="Model for optimization guidance (e.g., gpt-4o)",
    )
    compile_parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Path to a custom optimizer configuration YAML file. "
        "If not provided, it will look for 'settings/optimization.yaml' "
        "in the current directory.",
    )
    compile_parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save compiled artifact (JSON format)",
    )

    # Optional arguments
    compile_parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help='JSON string with additional model parameters (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )
    compile_parser.add_argument(
        "--program-style",
        choices=["cot", "predict"],
        default="cot",
        help="DSPy module style to use when building the program (cot or predict).",
    )
    compile_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    compile_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)",
    )
    compile_parser.add_argument(
        "--quiet", action="store_true", help="Disable verbose output"
    )
    compile_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    compile_parser.add_argument(
        "--k-window-size",
        type=int,
        default=None,
        help="Context window size (k) used for the trainset. Stores it as metadata in the artifact.",
    )

    # Schema subcommand
    schema_parser = subparsers.add_parser(
        "schema",
        help="Show expected JSONL schema for a component",
        description="Display the expected dataset schema for a specific component",
    )
    schema_parser.add_argument(
        "--component",
        choices=["selection", "disambiguation", "decomposition"],
        required=True,
        help="Component to show schema for",
    )

    # Generate dataset subcommand
    generate_parser = subparsers.add_parser(
        "generate-dataset",
        help="Generate a gold standard dataset using a teacher model",
        description="Create training datasets by using a powerful teacher model to generate structured outputs from raw text inputs",
    )
    generate_parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to input text file (one sentence per line) or a directory of text files.",
    )
    generate_parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output JSONL file",
    )
    generate_parser.add_argument(
        "--component",
        choices=["selection", "disambiguation", "decomposition"],
        required=True,
        help="Component to generate data for",
    )
    generate_parser.add_argument(
        "--teacher-model",
        required=True,
        help="Powerful teacher model to use for generation (e.g., gpt-4o, claude-3-opus)",
    )
    # Optional arguments
    generate_parser.add_argument(
        "--curated",
        action="store_true",
        help="Indicates that the input path is a directory of curated JSON files.",
    )
    generate_parser.add_argument(
        "--clean-markdown",
        action="store_true",
        help="Clean markdown syntax from input files before processing.",
    )
    generate_parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help='JSON string with additional model parameters (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )
    generate_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests to the teacher model.",
    )
    generate_parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Also generate curated negative samples (disambiguation & decomposition only).",
    )
    generate_parser.add_argument(
        "--negative-quota",
        type=int,
        default=0,
        help="Minimum count per failure mode for generated negatives (requires --include-negatives).",
    )
    generate_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit total teacher calls for smoke tests.",
    )

    return parser


def validate_compile_args(args: argparse.Namespace) -> None:
    """Validate arguments for the compile command.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check trainset file exists
    if not args.trainset.exists():
        print(
            f"❌ Error: Training dataset file not found: {args.trainset}",
            file=sys.stderr,
        )
        print(
            "💡 Hint: Make sure the file path is correct and the file exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check trainset is a file (not directory)
    if not args.trainset.is_file():
        print(
            f"❌ Error: Training dataset path is not a file: {args.trainset}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check config file exists if provided
    if args.config and not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}", file=sys.stderr)
        print(
            "💡 Hint: Make sure the file path is correct and the file exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check config is a file (not directory) if provided
    if args.config and not args.config.is_file():
        print(f"❌ Error: Config path is not a file: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Check output directory exists or can be created
    output_dir = args.output_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            f"❌ Error: Cannot create output directory {output_dir}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if output file exists and handle overwrite
    if args.output_path.exists() and not args.overwrite:
        print(
            f"❌ Error: Output file already exists: {args.output_path}", file=sys.stderr
        )
        print("💡 Hint: Use --overwrite to replace the existing file", file=sys.stderr)
        sys.exit(1)

    # Validate numeric arguments
    if args.seed is not None and args.seed < 0:
        print(f"❌ Error: Seed must be non-negative, got: {args.seed}", file=sys.stderr)
        sys.exit(1)


def validate_generate_args(args: argparse.Namespace) -> None:
    """Validate arguments for the generate-dataset command.

    Args:
        args: Parsed command line arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check input file exists
    if not args.input_path.exists():
        print(
            f"❌ Error: Input path not found: {args.input_path}",
            file=sys.stderr,
        )
        print(
            "💡 Hint: Make sure the file path is correct and the file exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check output directory exists or can be created
    output_dir = args.output_file.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            f"❌ Error: Cannot create output directory {output_dir}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check if output file exists
    if args.output_file.exists():
        print(
            f"❌ Error: Output file already exists: {args.output_file}", file=sys.stderr
        )
        print(
            "💡 Hint: Remove the existing file or specify a different output path",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.include_negatives and args.component == "selection":
        print(
            "ℹ️  Ignoring --include-negatives for the selection component (negatives already implicit in labels).",
            file=sys.stderr,
        )
        args.include_negatives = False

    if args.negative_quota < 0:
        print(
            f"❌ Error: --negative-quota must be >= 0, got {args.negative_quota}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.negative_quota and not args.include_negatives:
        print(
            "ℹ️  --negative-quota specified without --include-negatives; quota will be ignored.",
            file=sys.stderr,
        )
        args.negative_quota = 0

    if args.max_examples is not None and args.max_examples <= 0:
        print(
            "❌ Error: --max-examples must be a positive integer.",
            file=sys.stderr,
        )
        sys.exit(1)

    if hasattr(args, "concurrency") and args.concurrency <= 0:
        print(
            f"❌ Error: --concurrency must be a positive integer, got: {args.concurrency}",
            file=sys.stderr,
        )
        sys.exit(1)


def handle_init_command(args: argparse.Namespace) -> None:
    """Handle the init subcommand.

    Args:
        args: Parsed command line arguments
    """
    dest_dir = Path.cwd() / "settings"
    if dest_dir.exists() and not args.overwrite:
        print(f"❌ Error: Directory '{dest_dir}' already exists.", file=sys.stderr)
        print("💡 Use --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)

    try:
        # Remove existing directory if overwrite is true
        if dest_dir.exists() and args.overwrite:
            print(f"Removing existing directory: {dest_dir}")
            shutil.rmtree(dest_dir)

        # Create the settings directory
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy files from package resources
        package_settings_path = "aclarai_claimify.settings"
        files_to_copy = ["config.yaml", "optimization.yaml"]

        print(f"Creating local settings directory at: {dest_dir}")
        for filename in files_to_copy:
            with resources.path(package_settings_path, filename) as src_path:
                dest_path = dest_dir / filename
                shutil.copy(src_path, dest_path)
                print(f"  - Created {dest_path}")

        print("\n✅ Initialization complete.")
        print("You can now edit the files in the 'settings' directory.")

    except Exception as e:
        print(f"\n💥 Unexpected error during init: {e}", file=sys.stderr)
        sys.exit(1)


def handle_compile_command(args: argparse.Namespace) -> None:
    """Handle the compile subcommand.

    Args:
        args: Parsed command line arguments
    """
    # Validate arguments
    validate_compile_args(args)

    # Set verbosity
    verbose = args.verbose and not args.quiet

    if verbose:
        print("🚀 Aclarai Claimify - DSPy Optimization Toolkit")
        print("=" * 50)

    try:
        # Determine config path
        if args.config:
            optim_config_path = args.config
        else:
            local_optim_config = Path.cwd() / "settings" / "optimization.yaml"
            optim_config_path = (
                local_optim_config if local_optim_config.exists() else None
            )

        # Load configurations
        claimify_config = load_claimify_config(
            override_path=str(Path.cwd() / "settings" / "config.yaml")
        )
        optim_config = load_optimization_config(override_path=str(optim_config_path))

        # Parse model_params if provided
        model_params = {}
        if hasattr(args, "model_params") and args.model_params:
            import json

            try:
                model_params = json.loads(args.model_params)
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON in --model-params: {e}", file=sys.stderr)
                sys.exit(1)

        if verbose:
            print(
                f"   Optimizer Config: {optim_config_path if optim_config_path else 'Using default'}"
            )

        # Run compilation
        compile_component(
            component=args.component,
            train_path=args.trainset,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            output_path=args.output_path,
            seed=args.seed,
            verbose=verbose,
            model_params=model_params,
            k_window_size=args.k_window_size,
            program_style=args.program_style,
            claimify_config=claimify_config,
            optimizer_config=optim_config,
        )

        if verbose:
            print("\n🎉 Success! Your optimized component is ready to use.")
            print(f"📁 Artifact saved: {args.output_path}")
            print("\n💡 Next steps:")
            print("   1. Load the artifact in your components:")
            print(
                "      from aclarai_claimify.optimization.artifacts import load_artifact"
            )
            print(f"      artifact = load_artifact(Path('{args.output_path}'))")
            print("   2. See the README for integration examples")

    except DataValidationError as e:
        print(f"\n❌ Dataset Validation Error: {e}", file=sys.stderr)
        print("\n💡 Dataset Schema Help:", file=sys.stderr)
        print_schema_help(args.component)
        sys.exit(1)

    except ModelConfigError as e:
        print(f"\n❌ Model Configuration Error: {e}", file=sys.stderr)
        if "api_key" in str(e).lower():
            print("\n💡 Quick fix:", file=sys.stderr)
            print("   Set your model API key in environment variables", file=sys.stderr)
            print("   (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)", file=sys.stderr)
        sys.exit(1)

    except DSPyVersionError as e:
        print(f"\n❌ DSPy Compatibility Error: {e}", file=sys.stderr)
        print("\n💡 Try updating DSPy:", file=sys.stderr)
        print("   pip install --upgrade dspy-ai", file=sys.stderr)
        sys.exit(1)

    except OptimizationError as e:
        print(f"\n❌ Optimization Error: {e}", file=sys.stderr)
        print("\n💡 Common solutions:", file=sys.stderr)
        print("   1. Check your internet connection", file=sys.stderr)
        print("   2. Verify your API key has sufficient credits", file=sys.stderr)
        print("   3. Try with a smaller dataset or fewer trials", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Compilation interrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        print("\n💡 This might be a bug. Please report it with:", file=sys.stderr)
        print("   - Your command line arguments", file=sys.stderr)
        print("   - The error message above", file=sys.stderr)
        print("   - Your Python and DSPy versions", file=sys.stderr)
        sys.exit(1)


def handle_schema_command(args: argparse.Namespace) -> None:
    """Handle the schema subcommand.

    Args:
        args: Parsed command line arguments
    """
    print(f"📋 Dataset Schema for {args.component.title()} Component")
    print("=" * 60)
    print_schema_help(args.component)


def handle_generate_command(args: argparse.Namespace) -> None:
    """Handle the generate-dataset subcommand.

    Args:
        args: Parsed command line arguments
    """
    # Validate arguments
    validate_generate_args(args)

    # Load configuration
    claimify_config = load_claimify_config(
        override_path=str(Path.cwd() / "settings" / "config.yaml")
    )

    # Parse model_params if provided
    model_params = {}
    if hasattr(args, "model_params") and args.model_params:
        import json

        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as e:
            print(
                f"\n❌ Invalid JSON in --model-params: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("🚀 Aclarai Claimify - Gold Standard Dataset Generation")
    print("=" * 55)

    try:
        # Run generation
        generate_dataset(
            input_path=args.input_path,
            output_file=args.output_file,
            component=args.component,
            teacher_model=args.teacher_model,
            model_params=model_params,
            claimify_config=claimify_config,
            clean_markdown_flag=args.clean_markdown,
            curated_flag=args.curated,
            concurrency=args.concurrency,
            include_negatives=args.include_negatives,
            negative_quota=args.negative_quota,
            max_examples=args.max_examples,
        )

        print("\n🎉 Success! Your gold standard dataset is ready.")
        print(f"📁 Dataset saved: {args.output_file}")
        print("\n💡 Next steps:")
        print("   1. Review the generated dataset for quality")
        print("   2. Use it to compile your component:")
        print("      aclarai-claimify compile \\")
        print(f"          --component {args.component} \\")
        print(f"          --trainset {args.output_file} \\")
        print("          --student-model <your-student-model> \\")
        print(f"          --teacher-model {args.teacher_model} \\")
        print("          --config <optimizer-config.yaml> \\")
        print("          --output-path <compiled-artifact.json>")

    except GenerationError as e:
        print(f"\n❌ Dataset Generation Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        print("\n💡 This might be a bug. Please report it with:", file=sys.stderr)
        print("   - Your command line arguments", file=sys.stderr)
        print("   - The error message above", file=sys.stderr)
        print("   - Your Python and DSPy versions", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Route to appropriate handler
    if args.command == "init":
        handle_init_command(args)
    elif args.command == "compile":
        handle_compile_command(args)
    elif args.command == "schema":
        handle_schema_command(args)
    elif args.command == "generate-dataset":
        handle_generate_command(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
