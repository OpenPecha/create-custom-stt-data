#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Garchen Rinpoche STT Benchmark Data Sampler

This script creates a well-distributed benchmark dataset for speech recognition evaluation.
It uses stratified sampling to ensure the benchmark represents various audio conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import re
from collections import Counter, defaultdict

# Configuration
SEGMENTS_PER_STRATUM = 130   # Target number of segments to select from each stratum
MIN_SEGMENT_DURATION = 0.5  # Minimum duration in seconds for segments to be considered
MAX_SEGMENT_DURATION = 30.0 # Maximum duration in seconds for segments to be considered
RANDOM_SEED = 42  # For reproducibility
OUTPUT_DIR = "benchmark"

def parse_duration(time_str):
    """Convert time string (HH:MM:SS) to seconds"""
    if pd.isna(time_str):
        return 0
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def duration_category(seconds):
    """Categorize duration into short, medium, long"""
    if seconds < 300:  # Less than 5 minutes
        return "short"
    elif seconds < 900:  # Less than 15 minutes
        return "medium"
    else:
        return "long"

def extract_content_type(filename):
    """Extract content type from filename"""
    lower_name = filename.lower()
    if "qa" in lower_name:
        return "Q&A"
    elif "practice" in lower_name:
        return "Practice"
    elif "dedication" in lower_name or "prayer" in lower_name:
        return "Prayer"
    else:
        return "Teaching"

def extract_original_id(segment_id):
    """Extract original file ID from a segment ID
    e.g., STT_GR_0001_0003_22300_to_27800 -> STT_GR_0001
    """
    match = re.match(r'(STT_GR_\d+)_', segment_id)
    if match:
        return match.group(1)
    return segment_id

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read catalog data (original files)
    catalog_df = pd.read_csv("gr_catalog.csv")
    
    # Read split segments data
    segments_df = pd.read_csv("GR.csv")
    
    print(f"Original catalog contains {len(catalog_df)} audio files")
    print(f"Split segments dataset contains {len(segments_df)} segments")
    
    # Filter segments by duration
    segments_df['audio_duration'] = segments_df['audio_duration'].astype(float)
    valid_segments = segments_df[
        (segments_df['audio_duration'] >= MIN_SEGMENT_DURATION) & 
        (segments_df['audio_duration'] <= MAX_SEGMENT_DURATION)
    ]
    print(f"Found {len(valid_segments)} segments within duration range {MIN_SEGMENT_DURATION}-{MAX_SEGMENT_DURATION} seconds")
    
    # Add original file ID to segments
    valid_segments['original_id'] = valid_segments['file_name'].apply(extract_original_id)
    
    # Add derived metrics to catalog
    catalog_df['split_duration_seconds'] = catalog_df['Original Full Audio Duration'].apply(parse_duration)
    catalog_df['duration_category'] = catalog_df['split_duration_seconds'].apply(duration_category)
    catalog_df['content_type'] = catalog_df['Filename'].apply(extract_content_type)
    
    # Create strata in catalog
    catalog_df['strata'] = catalog_df.apply(
        lambda row: f"{row['AGE']}__{row['duration_category']}__{row['content_type']}",
        axis=1
    )
    
    # Create ID to strata mapping
    id_to_strata = dict(zip(catalog_df['ID'], catalog_df['strata']))
    id_to_metadata = catalog_df.set_index('ID')[['AGE', 'duration_category', 'content_type']].to_dict('index')
    
    # Map segments to strata
    valid_segments['strata'] = valid_segments['original_id'].map(id_to_strata)
    
    # Drop segments without strata (meaning they don't match any original file)
    valid_segments = valid_segments.dropna(subset=['strata'])
    print(f"Found {len(valid_segments)} segments with valid strata information")
    
    # Count unique strata
    strata_counts = Counter(valid_segments['strata'])
    print(f"\nFound {len(strata_counts)} unique strata combinations")
    print("Strata distribution:")
    for stratum, count in strata_counts.items():
        print(f"  {stratum}: {count} segments")
    
    # Select benchmark segments using stratified sampling
    np.random.seed(RANDOM_SEED)
    benchmark_segments = []
    
    for stratum, count in strata_counts.items():
        stratum_segments = valid_segments[valid_segments['strata'] == stratum]
        
        # Determine how many segments to select from this stratum
        target_count = min(SEGMENTS_PER_STRATUM, len(stratum_segments))
        
        # Sample randomly from this stratum
        if len(stratum_segments) <= target_count:
            # Take all if we have fewer than target
            selected = stratum_segments
        else:
            # Sample the target count
            selected = stratum_segments.sample(n=target_count)
            
        benchmark_segments.append(selected)
    
    # Combine all selected segments
    benchmark_df = pd.concat(benchmark_segments)
    
    # Add metadata from original files
    benchmark_df['age_group'] = benchmark_df['original_id'].map(lambda x: id_to_metadata.get(x, {}).get('AGE', 'Unknown'))
    benchmark_df['duration_category'] = benchmark_df['original_id'].map(lambda x: id_to_metadata.get(x, {}).get('duration_category', 'Unknown'))
    benchmark_df['content_type'] = benchmark_df['original_id'].map(lambda x: id_to_metadata.get(x, {}).get('content_type', 'Unknown'))
    
    print(f"\nSelected {len(benchmark_df)} benchmark segments from {len(strata_counts)} strata")
    
    # Save the original benchmark segments
    benchmark_df.to_csv(f"{OUTPUT_DIR}/benchmark_segments.csv", index=False)
    
    # Create a new DataFrame for the requested format
    formatted_df = benchmark_df.copy()
    
    # Add the requested columns for the formatted output
    # Add group_id (33)
    formatted_df['group_id'] = 33
    # Add state (transcribing)
    formatted_df['state'] = 'transcribing'
    # Add id (2159570 incremented by 1)
    formatted_df['id'] = range(2159570, 2159570 + len(formatted_df))
    
    # Ensure inference_transcript column exists
    if 'inference_transcript' not in formatted_df.columns:
        formatted_df['inference_transcript'] = ''
    
    # Select only the columns we need in the specified order
    columns_order = [
        'file_name', 'url', 'inference_transcript', 'audio_duration', 
        'group_id', 'state', 'id'
    ]
    formatted_output = formatted_df[columns_order]
    
    # Save the formatted benchmark segments to a new file
    formatted_output.to_csv(f"{OUTPUT_DIR}/benchmark_for_upload.csv", index=False)
    print(f"Created formatted benchmark file at {OUTPUT_DIR}/benchmark_for_upload.csv")
    
    # Save mapping of all segments to their strata
    strata_mapping_df = valid_segments[['file_name', 'original_id', 'strata', 'audio_duration']]
    strata_mapping_df.to_csv(f"{OUTPUT_DIR}/segment_strata_mapping.csv", index=False)
    
    # Save original full audio files with their strata
    full_audio_strata_df = catalog_df[['ID', 'Filename', 'strata', 'AGE', 'duration_category', 'content_type', 'split_duration_seconds']]
    full_audio_strata_df.to_csv(f"{OUTPUT_DIR}/full_audio_strata.csv", index=False)
    print(f"Saved strata information for {len(full_audio_strata_df)} full audio files to {OUTPUT_DIR}/full_audio_strata.csv")
    
    # Generate JSON format suitable for annotation
    benchmark_json = []
    for _, row in benchmark_df.iterrows():
        benchmark_json.append({
            "id": row['file_name'],
            "url": row['url'],
            "duration": float(row['audio_duration']),
            "metadata": {
                "original_file_id": row['original_id'],
                "age_group": row['age_group'],
                "content_type": row['content_type'],
                "duration_category": row['duration_category'],
                "strata": row['strata']
            },
            "segment_info": {
                "transcript": row['inference_transcript'] if 'inference_transcript' in row.index else ""
            }
        })
    
    # Save JSON for annotation
    with open(f"{OUTPUT_DIR}/benchmark_annotation.json", "w") as f:
        json.dump(benchmark_json, f, indent=2)
    
    # Generate report on benchmark distribution
    print("\nBenchmark Distribution:")
    print(f"Age groups: {benchmark_df['age_group'].value_counts().to_dict()}")
    print(f"Duration categories: {benchmark_df['duration_category'].value_counts().to_dict()}")
    print(f"Content types: {benchmark_df['content_type'].value_counts().to_dict()}")
    print(f"Average segment duration: {benchmark_df['audio_duration'].mean():.2f} seconds")
    
    # Save a report
    with open(f"{OUTPUT_DIR}/benchmark_report.md", "w") as f:
        f.write("# Garchen Rinpoche STT Benchmark Dataset Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total catalog size: {len(catalog_df)} original audio files\n")
        f.write(f"Total segments: {len(segments_df)} audio segments\n")
        f.write(f"Valid segments (duration filtered): {len(valid_segments)} audio segments\n")
        f.write(f"Benchmark size: {len(benchmark_df)} audio segments\n\n")
        
        f.write("## Sampling Strategy\n\n")
        f.write("Stratified sampling was used to ensure representation across:\n")
        f.write("- Age groups\n")
        f.write("- Duration categories\n")
        f.write("- Content types\n\n")
        f.write(f"Target segments per stratum: {SEGMENTS_PER_STRATUM}\n")
        f.write(f"Duration filter range: {MIN_SEGMENT_DURATION}-{MAX_SEGMENT_DURATION} seconds\n\n")
        
        f.write("## Distribution of Strata in Original Dataset\n\n")
        f.write("### Age Groups\n")
        for age, count in catalog_df['AGE'].value_counts().items():
            f.write(f"- {age}: {count} ({count/len(catalog_df)*100:.1f}%)\n")
        
        f.write("\n### Duration Categories\n")
        for cat, count in catalog_df['duration_category'].value_counts().items():
            f.write(f"- {cat}: {count} ({count/len(catalog_df)*100:.1f}%)\n")
        
        f.write("\n### Content Types\n")
        for type_, count in catalog_df['content_type'].value_counts().items():
            f.write(f"- {type_}: {count} ({count/len(catalog_df)*100:.1f}%)\n")
        
        f.write("\n## Distribution in Valid Segments\n\n")
        # Count occurrences of each stratum in valid segments
        strata_distribution = valid_segments['strata'].value_counts()
        f.write(f"Total unique strata: {len(strata_distribution)}\n\n")
        for stratum, count in strata_distribution.items():
            f.write(f"- {stratum}: {count} segments ({count/len(valid_segments)*100:.1f}%)\n")
        
        f.write("\n## Distribution in Benchmark Selection\n\n")
        f.write("### Age Groups\n")
        for age, count in benchmark_df['age_group'].value_counts().items():
            f.write(f"- {age}: {count} ({count/len(benchmark_df)*100:.1f}%)\n")
        
        f.write("\n### Duration Categories\n")
        for cat, count in benchmark_df['duration_category'].value_counts().items():
            f.write(f"- {cat}: {count} ({count/len(benchmark_df)*100:.1f}%)\n")
        
        f.write("\n### Content Types\n")
        for type_, count in benchmark_df['content_type'].value_counts().items():
            f.write(f"- {type_}: {count} ({count/len(benchmark_df)*100:.1f}%)\n")
        
        # Add segment duration statistics
        f.write("\n### Segment Duration Statistics\n")
        f.write(f"- Average duration: {benchmark_df['audio_duration'].mean():.2f} seconds\n")
        f.write(f"- Minimum duration: {benchmark_df['audio_duration'].min():.2f} seconds\n")
        f.write(f"- Maximum duration: {benchmark_df['audio_duration'].max():.2f} seconds\n")

if __name__ == "__main__":
    main()
    print("\nBenchmark sampling completed. Results saved to the 'benchmark' directory.")
