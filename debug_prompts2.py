#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/Shared/Public/Github/aclarai_claimify')

from aclarai_claimify.optimization.generate import TEACHER_PROMPTS

def test_prompts():
    print("Examining selection prompt...")
    selection_prompt = TEACHER_PROMPTS["selection"]
    print(f"Selection prompt length: {len(selection_prompt)}")
    print(f"Selection prompt preview: {repr(selection_prompt[:200])}")
    
    # Look for format placeholders
    import re
    placeholders = re.findall(r'\{[^}]+\}', selection_prompt)
    print(f"Format placeholders found: {placeholders}")
    
    # Look for JSON-like strings
    json_strings = re.findall(r'"[^"]*"', selection_prompt)
    print(f"JSON-like strings found: {json_strings[:5]}...")  # Show first 5
    
    print("\n" + "="*50 + "\n")
    
    print("Examining disambiguation prompt...")
    disambiguation_prompt = TEACHER_PROMPTS["disambiguation"]
    print(f"Disambiguation prompt length: {len(disambiguation_prompt)}")
    print(f"Disambiguation prompt preview: {repr(disambiguation_prompt[:200])}")
    
    # Look for format placeholders
    placeholders = re.findall(r'\{[^}]+\}', disambiguation_prompt)
    print(f"Format placeholders found: {placeholders}")
    
    # Look for JSON-like strings
    json_strings = re.findall(r'"[^"]*"', disambiguation_prompt)
    print(f"JSON-like strings found: {json_strings[:5]}...")  # Show first 5

if __name__ == "__main__":
    test_prompts()