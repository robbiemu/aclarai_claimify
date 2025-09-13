#!/usr/bin/env python3
import re
import sys
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
try:
    import requests
except ImportError:
    print("The 'requests' library is not installed. Please install it with 'pip install requests'")
    sys.exit(1)


def check_pedigree_robots_txt(pedigree_file="PEDIGREE.md", user_agent="*"):
    """
    Parses a PEDIGREE.md file, extracts non-synthetic URLs, and checks
    if they are allowed to be fetched according to the site's robots.txt.
    Includes a progress bar for user feedback.

    Args:
        pedigree_file (str): The path to the PEDIGREE.md file.
        user_agent (str): The user-agent to check against in robots.txt.
    """
    try:
        with open(pedigree_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{pedigree_file}' was not found.")
        return

    entries = content.split("---")
    
    # First pass: find all relevant entries to get a total count
    research_entries = []
    for entry in entries:
        if re.search(r"Source Type:.*?'?researched'?", entry, re.IGNORECASE) and not re.search(r"Source Type:.*?'?synthetic'?", entry, re.IGNORECASE):
            research_entries.append(entry)

    total_entries = len(research_entries)
    if total_entries == 0:
        print(f"No non-synthetic entries found in '{pedigree_file}'.")
        return

    # Print the total count and set up the progress bar
    print(f"{total_entries}", end="", flush=True)

    errant_entries = []
    processed_count = 0
    hashes_printed = 0
    
    # Second pass: process the entries and update the progress bar
    for entry in research_entries:
        processed_count += 1
        url_match = re.search(r"Source URL.*?[:\s`\(](https?://[^\s`\)]+)", entry, re.IGNORECASE)
        
        if url_match:
            url = url_match.group(1).strip()
            if url.endswith(']'):
                url = url[:-1]
            
            try:
                parsed_url = urlparse(url)
                robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                if not rp.can_fetch(user_agent, url):
                    errant_entries.append(url)
            except Exception:
                # Silently ignore errors during the check to not disrupt the progress bar
                pass

        # Update the progress bar
        progress_percentage = (processed_count / total_entries) * 100
        hashes_to_print = int(progress_percentage // 3)
        
        if hashes_to_print > hashes_printed:
            print("#" * (hashes_to_print - hashes_printed), end="", flush=True)
            hashes_printed = hashes_to_print
    
    # Finalize the output
    print("\n") # Move to the next line after the progress bar
    
    print(f"Checked {total_entries} non-synthetic entries from '{pedigree_file}'.")
    if errant_entries:
        print("\nErrant entries (disallowed by robots.txt):")
        for url in errant_entries:
            print(f"- {url}")
    else:
        print("No errant entries found.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "PEDIGREE.md"
    
    check_pedigree_robots_txt(filepath)
