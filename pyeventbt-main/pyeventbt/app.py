"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

import argparse
import sys
from . import __version__


def main():
    """
    Main entry point for the PyEventBT CLI.
    """
    parser = argparse.ArgumentParser(
        description="PyEventBT - Event-Driven Backtesting Framework"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"PyEventBT {__version__}"
    )
    parser.add_argument(
        "command", 
        nargs="?", 
        choices=["info"], 
        help="Command to run (default: info)"
    )

    args = parser.parse_args()

    # Default behavior or 'info' command
    if args.command == "info" or args.command is None:
        print_info()


def print_info():
    print(f"PyEventBT v{__version__}")
    print("========================================")
    print("Documentation: https://pyeventbt.com")
    print("GitHub:        https://github.com/marticastany/pyeventbt")
    print("========================================")
    print("\nPyEventBT is a framework/library.")
    print("To use it in your project, import the components:")
    print("\n    from pyeventbt import Strategy, BarEvent, SignalEvent")
    print("\nFor examples and tutorials, visit the documentation.")


if __name__ == "__main__":
    main()
