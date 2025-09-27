"""This script verifies the integrity of the curated prospect files.

It iterates through the prospect directories for each component, checks for empty or invalid JSON files, and reports the number of invalid files for each component.
It can also compose a list of the original source files that correspond to the invalid prospect files, which can be used to rerun the prospector on the failed files.
"""

import argparse
import glob
import json
import os
import sys


def main():
    """Main entry point for the verification script.
    Parses command-line arguments and orchestrates the verification process.
    """
    parser = argparse.ArgumentParser(description="Verify the curated prospect files.")
    parser.add_argument("prospects_dir", help="The root directory of the prospects.")
    parser.add_argument("--compose", help="Compose a list of source files to rerun.")
    parser.add_argument(
        "--component",
        choices=["selection", "disambiguation", "decomposition"],
        help="The component to verify.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.prospects_dir):
        print(
            f"Error: Prospects directory not found: {args.prospects_dir}",
            file=sys.stderr,
        )
        return

    components_to_verify = (
        [args.component]
        if args.component
        else ["selection", "disambiguation", "decomposition"]
    )

    rerun_files = []
    for component in components_to_verify:
        component_dir = os.path.join(args.prospects_dir, component)
        if not os.path.isdir(component_dir):
            print(
                f"Warning: Directory not found for component '{component}': {component_dir}",
                file=sys.stderr,
            )
            continue

        invalid_files_count = 0
        files = glob.glob(os.path.join(component_dir, "*.json"))
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                if not data or data == "" or data == {}:
                    invalid_files_count += 1
                    if args.compose and component == args.component:
                        # Construct the path to the original source file
                        source_filename = os.path.splitext(os.path.basename(file_path))[
                            0
                        ]
                        source_file_path = os.path.join(
                            "examples/data/datasets/tier1",
                            component,
                            source_filename + ".md",
                        )
                        rerun_files.append(source_file_path)
            except json.JSONDecodeError:
                invalid_files_count += 1
                if args.compose and component == args.component:
                    # Construct the path to the original source file
                    source_filename = os.path.splitext(os.path.basename(file_path))[0]
                    source_file_path = os.path.join(
                        "examples/data/datasets/tier1",
                        component,
                        source_filename + ".md",
                    )
                    rerun_files.append(source_file_path)

        print(f"Component: {component}")
        print(f"  - Total files: {len(files)}")
        print(f"  - Invalid files: {invalid_files_count}")

    if args.compose:
        with open(args.compose, "w", encoding="utf-8") as f_out:
            for file_path in rerun_files:
                f_out.write(file_path + "\n")
        print(
            f"\nSuccessfully composed list of {len(rerun_files)} files to rerun in '{args.compose}'"
        )


if __name__ == "__main__":
    main()
