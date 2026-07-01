"""
colors.py

Terminal color helpers for debug output.
Import gprint, rprint, yprint in any file that needs colored prints.

Convention:
    gprint — green   — node prints (permanent)
    yprint — yellow  — graph/routing prints (permanent)
    bprint — blue    — tool calls (permanent)
    rprint — red     — temporary debug (remove before commit)
"""

def gprint(text): print(f"\033[92m[NODE]  {text}\033[0m")
def yprint(text): print(f"\033[93m[ROUTE] {text}\033[0m")
def bprint(text): print(f"\033[94m[TOOL]  {text}\033[0m")
def rprint(text): print(f"\033[91m[DEBUG] {text}\033[0m")