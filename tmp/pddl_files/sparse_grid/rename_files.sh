#!/bin/bash

echo "Start to rename files..."

for f in CELL_*.bddl; do
  # Check if files exist, in case there is no matching items
  if [ -f "$f" ]; then
    new_name="$(echo "$f" | cut -d'_' -f1-3).bddl"
    echo "Rename '$f' as '$new_name'"
    # Comment the mv command, and make sure names are correct before executing
    mv -- "$f" "$new_name"
  fi
done

echo "Done."
