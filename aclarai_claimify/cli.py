"""Command-line interface for the DSPy optimization toolkit.

This module provides the CLI for compiling Claimify components using DSPy,
allowing users to optimize their own datasets and models.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

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
except ImportError as e:

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
        required=True,
        help="Path to optimizer configuration YAML file",
    )
    compile_parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save compiled artifact (JSON format)",
    )

    # Optional arguments
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
        "--input-file",
        type=Path,
        required=True,
        help="Path to input text file (one sentence per line)",
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
            f"💡 Hint: Make sure the file path is correct and the file exists",
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

    # Check config file exists
    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}", file=sys.stderr)
        print(
            f"💡 Hint: Make sure the file path is correct and the file exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check config is a file (not directory)
    if not args.config.is_file():
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
        print(f"💡 Hint: Use --overwrite to replace the existing file", file=sys.stderr)
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
    if not args.input_file.exists():
        print(
            f"❌ Error: Input file not found: {args.input_file}",
            file=sys.stderr,
        )
        print(
            f"💡 Hint: Make sure the file path is correct and the file exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check input file is a file (not directory)
    if not args.input_file.is_file():
        print(
            f"❌ Error: Input path is not a file: {args.input_file}",
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
        print(f"💡 Hint: Remove the existing file or specify a different output path", file=sys.stderr)
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
        # Run compilation
        compile_component(
            component=args.component,
            train_path=args.trainset,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            config_path=args.config,
            output_path=args.output_path,
            seed=args.seed,
            verbose=verbose,
        )

        if verbose:
            print("\n🎉 Success! Your optimized component is ready to use.")
            print(f"📁 Artifact saved: {args.output_path}")
            print("\n💡 Next steps:")
            print(f"   1. Load the artifact in your components:")
            print(
                f"      from aclarai_claimify.optimization.artifacts import load_artifact"
            )
            print(f"      artifact = load_artifact(Path('{args.output_path}'))")
            print(f"   2. See the README for integration examples")

    except DataValidationError as e:
        print(f"\n❌ Dataset Validation Error: {e}", file=sys.stderr)
        print(f"\n💡 Dataset Schema Help:", file=sys.stderr)
        print_schema_help(args.component)
        sys.exit(1)

    except ModelConfigError as e:
        print(f"\n❌ Model Configuration Error: {e}", file=sys.stderr)
        if "api_key" in str(e).lower():
            print(f"\n💡 Quick fix:", file=sys.stderr)
            print(f"   export OPENAI_API_KEY=your-api-key-here", file=sys.stderr)
        sys.exit(1)

    except DSPyVersionError as e:
        print(f"\n❌ DSPy Compatibility Error: {e}", file=sys.stderr)
        print(f"\n💡 Try updating DSPy:", file=sys.stderr)
        print(f"   pip install --upgrade dspy-ai", file=sys.stderr)
        sys.exit(1)

    except OptimizationError as e:
        print(f"\n❌ Optimization Error: {e}", file=sys.stderr)
        print(f"\n💡 Common solutions:", file=sys.stderr)
        print(f"   1. Check your internet connection", file=sys.stderr)
        print(f"   2. Verify your API key has sufficient credits", file=sys.stderr)
        print(f"   3. Try with a smaller dataset or fewer trials", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️  Compilation interrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        print(f"\n💡 This might be a bug. Please report it with:", file=sys.stderr)
        print(f"   - Your command line arguments", file=sys.stderr)
        print(f"   - The error message above", file=sys.stderr)
        print(f"   - Your Python and DSPy versions", file=sys.stderr)
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

    print("🚀 Aclarai Claimify - Gold Standard Dataset Generation")
    print("=" * 55)

    try:
        # Run generation
        generate_dataset(
            input_file=args.input_file,
            output_file=args.output_file,
            component=args.component,
            teacher_model=args.teacher_model,
        )

        print("\n🎉 Success! Your gold standard dataset is ready.")
        print(f"📁 Dataset saved: {args.output_file}")
        print("\n💡 Next steps:")
        print(f"   1. Review the generated dataset for quality")
        print(f"   2. Use it to compile your component:")
        print(f"      aclarai-claimify compile \\")
        print(f"          --component {args.component} \\")
        print(f"          --trainset {args.output_file} \\")
        print(f"          --student-model <your-student-model> \\")
        print(f"          --teacher-model {args.teacher_model} \\")
        print(f"          --config <optimizer-config.yaml> \\")
        print(f"          --output-path <compiled-artifact.json>")

    except GenerationError as e:
        print(f"\n❌ Dataset Generation Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️  Generation interrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", file=sys.stderr)
        print(f"\n💡 This might be a bug. Please report it with:", file=sys.stderr)
        print(f"   - Your command line arguments", file=sys.stderr)
        print(f"   - The error message above", file=sys.stderr)
        print(f"   - Your Python and DSPy versions", file=sys.stderr)
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
    if args.command == "compile":
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
