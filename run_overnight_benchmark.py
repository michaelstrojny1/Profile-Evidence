#!/usr/bin/env python3
"""
OVERNIGHT BENCHMARK LAUNCHER
===========================

This script launches the optimized overnight benchmark with live monitoring.
It ensures all compensation detection requirements are met and provides
comprehensive validation of PE's correction of PL's underestimation.

Usage:
    python run_overnight_benchmark.py

This will:
1. Launch overnight_benchmark_optimized.py in background
2. Start live progress monitoring
3. Save all logs and results with timestamps
"""

import subprocess
import time
import os
from datetime import datetime

def launch_benchmark():
    """Launch the overnight benchmark with live monitoring."""
    print("=" * 80)
    print("OVERNIGHT BENCHMARK LAUNCHER")
    print("=" * 80)
    print(f"Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"overnight_benchmark_{timestamp}.log"
    
    print(f"Launching optimized benchmark...")
    print(f"Log file: {log_file}")
    print(f"Monitor: Press Ctrl+C to stop monitoring (benchmark continues)")
    print()
    
    # Launch benchmark in background
    cmd = [
        'python', '-u', 'overnight_benchmark_optimized.py'
    ]
    
    print("Starting benchmark process...")
    
    # Use PowerShell Tee-Object for live logging
    ps_cmd = f'python -u overnight_benchmark_optimized.py 2>&1 | Tee-Object -FilePath {log_file} -Append'
    
    process = subprocess.Popen(
        ['powershell', '-Command', ps_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print(f"Benchmark launched with PID: {process.pid}")
    print("=" * 80)
    print("LIVE PROGRESS MONITORING")
    print("=" * 80)
    
    try:
        # Monitor live output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        # Wait for completion
        process.wait()
        
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETED!")
        print("=" * 80)
        print(f"Exit code: {process.returncode}")
        print(f"Log saved: {log_file}")
        
        # Show final status
        if process.returncode == 0:
            print("Success: Benchmark completed successfully")
        else:
            print("✗ ERROR: Benchmark failed")
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("MONITORING STOPPED BY USER")
        print("=" * 80)
        print("Note: Benchmark continues running in background")
        print(f"Check log file for progress: {log_file}")
        print(f"Process PID: {process.pid}")
        
        # Don't kill the process, let it continue
        return process, log_file
    
    return process, log_file

if __name__ == '__main__':
    launch_benchmark()
