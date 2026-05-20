import sys
import uproot

if len(sys.argv) < 2:
    print("Usage: python inspect_qpix_root.py path/to/file.root")
    sys.exit(1)

fname = sys.argv[1]
f = uproot.open(fname)

print(f"\nFile: {fname}")
print("\n=== Top-level keys (TTrees, histos, etc.) ===")
for k in f.keys():
    print(" ", k)

print("\nIf you see something like 'tree;1' or 'Events;1', that's a TTree.")

# OPTIONAL: if you already know the tree name, uncomment & set it here:
tree = f["event_tree"]  # <-- replace with actual name
print("\nBranches in tree:")
print(tree.keys())