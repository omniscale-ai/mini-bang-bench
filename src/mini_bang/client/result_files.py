"""Old public import name, kept for agent workspaces and saved agent scripts.

The one document writer and reader is readable_files.py. Public images copy
this file next to it, so `from result_files import read_document` keeps working.
"""
try:
    from .readable_files import read_document, write_document
except ImportError:  # copied flat into a public agent image or workspace
    from readable_files import read_document, write_document
