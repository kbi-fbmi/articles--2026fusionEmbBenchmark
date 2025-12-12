#!/bin/bash
# filepath: compare_folders.sh

# Usage: ./compare_folders.sh <folder1> <folder2>
# For each file in folder1, compare with file of the same name in folder2.
# Only compare as many rows as the file in folder1 has.

folder1="$1"
folder2="$2"
echo "Comparing files in $folder1 with files in $folder2 ..."

for file1 in "$folder1"/*; do
    fname=$(basename "$file1")
    file2="$folder2/$fname"
    echo "Comparing $file1 with $file2"
    if [[ ! -f "$file2" ]]; then
        echo "File $fname not found in $folder2"
        continue
    fi
    
    nrows=$(wc -l < "$file1")
    echo "Comparing $fname (first $nrows rows)..."
    diff -bB --strip-trailing-cr <(head -n "$nrows" "$file1") <(head -n "$nrows" "$file2") >> /dev/null
    if [[ $? -eq 0 ]]; then
        echo "Files $fname are identical."
    else
        echo "Files $fname differ."
    fi
    echo "-----------------------------------"



done