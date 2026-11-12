"""
test_cec.py
Verifies the integration of the official CEC 2022 framework.
"""
import os
import sys

# Add the exact folder containing CEC2022.py to Python's system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cec2022', 'Python'))

try:
    # Import the function from the CEC2022.py file
    from CEC2022 import cec2022_func
    print("SUCCESS: The CEC 2022 framework is successfully linked!")
except ImportError as e:
    print("ERROR: Could not import.")
    print(e)
    print("\nTroubleshooting: Open cec2022/Python/CEC2022.py and check the exact name of the main evaluation function. Replace 'cec2022_func' in this script if it is named something else (e.g., 'cec22_test_func').")