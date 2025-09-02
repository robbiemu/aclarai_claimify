#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/Shared/Public/Github/aclarai_claimify')

from aclarai_claimify.optimization.generate import TEACHER_PROMPTS

def test_formatting():
    print("Testing selection prompt formatting...")
    try:
        prompt = TEACHER_PROMPTS["selection"].format(
            context_text="[0] The system was stable.",
            target_sentence="It failed with error code 500."
        )
        print("SUCCESS: Selection prompt formatted correctly")
        print(f"Prompt preview: {prompt[:200]}...")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting disambiguation prompt formatting...")
    try:
        prompt = TEACHER_PROMPTS["disambiguation"].format(
            context_text="[0] The system was stable.",
            target_sentence="It failed with error code 500."
        )
        print("SUCCESS: Disambiguation prompt formatted correctly")
        print(f"Prompt preview: {prompt[:200]}...")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting decomposition prompt formatting...")
    try:
        prompt = TEACHER_PROMPTS["decomposition"].format(
            disambiguated_text="The system failed with error code 500."
        )
        print("SUCCESS: Decomposition prompt formatted correctly")
        print(f"Prompt preview: {prompt[:200]}...")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_formatting()