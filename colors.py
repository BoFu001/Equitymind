"""
colors.py

Terminal color helpers for debug output.
Import gprint, rprint, yprint in any file that needs colored prints.

Convention:
    gprint — green   — node prints 
    yprint — yellow  — graph/routing prints 
    bprint — blue    — data fetch
    mprint — magenta — quant signal computation
    rprint — red     — temporary debug 
"""

def gprint(text): print(f"\033[92m[NODE] {text}\033[0m")
def yprint(text): print(f"\033[93m[ROUTE]{text}\033[0m")
def bprint(text): print(f"\033[94m[DATA] {text}\033[0m")
def mprint(text): print(f"\033[95m[QUANT]{text}\033[0m")
def rprint(text): print(f"\033[91m[DEBUG]{text}\033[0m")