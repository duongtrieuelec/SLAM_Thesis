#!/usr/bin/env python3
"""
Associate RGB and depth images from TUM RGB-D dataset.
Based on TUM's associate.py script.
"""

import argparse
import sys

def read_file_list(filename):
    """
    Reads a trajectory from a text file.
    Returns a dict of {timestamp: filename}.
    """
    file = open(filename)
    data = {}
    for line in file:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 2:
            timestamp = float(parts[0])
            data[timestamp] = parts[1]
    file.close()
    return data

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of timestamps.
    Returns a list of matched pairs [(t1, t2), ...].
    """
    matches = []
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                        for a in first_keys 
                        for b in second_keys 
                        if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    
    first_used = set()
    second_used = set()
    
    for diff, a, b in potential_matches:
        if a not in first_used and b not in second_used:
            matches.append((a, b))
            first_used.add(a)
            second_used.add(b)
    
    matches.sort()
    return matches

def main():
    parser = argparse.ArgumentParser(description='Associate RGB and depth images')
    parser.add_argument('first_file', help='First file (e.g., rgb.txt)')
    parser.add_argument('second_file', help='Second file (e.g., depth.txt)')
    parser.add_argument('--offset', type=float, default=0.0,
                       help='Time offset between files')
    parser.add_argument('--max_difference', type=float, default=0.02,
                       help='Maximum time difference for association')
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)

    matches = associate(first_list, second_list, args.offset, args.max_difference)

    for a, b in matches:
        print(f"{a:.6f} {first_list[a]} {b:.6f} {second_list[b]}")

if __name__ == '__main__':
    main()
