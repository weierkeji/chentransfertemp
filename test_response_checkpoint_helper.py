"""
Helper script for automated response checkpoint testing

This script provides utilities to:
1. Monitor training logs in real-time
2. Automatically interrupt training at specified progress points
3. Extract timing information from logs
4. Generate test reports

Usage:
    # Monitor training and interrupt at 50% progress
    python tests/test_response_checkpoint_helper.py monitor --log-file training.log --interrupt-at 50
    
    # Extract timing from logs
    python tests/test_response_checkpoint_helper.py extract-timing --log-file training.log
    
    # Create test dataset
    python tests/test_response_checkpoint_helper.py create-dataset --source <path> --output <path> --size 200
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class LogMonitor:
    """Monitor training logs and track progress"""
    
    def __init__(self, log_file: str, total_prompts: Optional[int] = None):
        self.log_file = log_file
        self.total_prompts = total_prompts
        self.completed_prompts = 0
        self.current_step = 0
        self.timing_data = {}
        
    def parse_log_line(self, line: str) -> Dict:
        """Parse a log line and extract relevant information"""
        result = {}
        
        # Extract ResponseCheckpoint messages
        if "[ResponseCheckpoint]" in line:
            if "Saved response" in line:
                # Extract query_id and response_idx
                match = re.search(r"query_id=([^,]+).*response_idx=(\d+)", line)
                if match:
                    result["type"] = "response_saved"
                    result["query_id"] = match.group(1)
                    result["response_idx"] = int(match.group(2))
            
            elif "Prompt completed" in line:
                # Extract completed prompts
                match = re.search(r"completed_prompts=(\d+)/(\d+)", line)
                if match:
                    result["type"] = "prompt_completed"
                    result["completed"] = int(match.group(1))
                    result["total"] = int(match.group(2))
                    self.completed_prompts = result["completed"]
                    if self.total_prompts is None:
                        self.total_prompts = result["total"]
            
            elif "Loaded checkpoint" in line:
                match = re.search(r"step (\d+).*completed_prompts=(\d+)", line)
                if match:
                    result["type"] = "checkpoint_loaded"
                    result["step"] = int(match.group(1))
                    result["completed_prompts"] = int(match.group(2))
        
        # Extract timing information
        if "rollout_step" in line and "took" in line:
            match = re.search(r"rollout_step.*took\s+([\d.]+)", line)
            if match:
                result["type"] = "timing"
                result["phase"] = "rollout_step"
                result["time"] = float(match.group(1))
                self.timing_data["rollout_step"] = result["time"]
        
        return result
    
    def get_progress_percentage(self) -> float:
        """Get current progress as percentage"""
        if self.total_prompts and self.total_prompts > 0:
            return (self.completed_prompts / self.total_prompts) * 100
        return 0.0
    
    def tail_log(self, callback=None):
        """Tail the log file and yield parsed lines"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            # Start from beginning
            f.seek(0)
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                
                parsed = self.parse_log_line(line)
                if parsed and callback:
                    callback(parsed, line)
                
                yield line, parsed


class TrainingInterrupter:
    """Automatically interrupt training at specified progress"""
    
    def __init__(self, process_pid: int, interrupt_at_percent: float):
        self.process_pid = process_pid
        self.interrupt_at_percent = interrupt_at_percent
        self.interrupted = False
    
    def check_and_interrupt(self, progress_percent: float):
        """Check if we should interrupt and do so if needed"""
        if not self.interrupted and progress_percent >= self.interrupt_at_percent:
            print(f"\n[INTERRUPT] Progress {progress_percent:.1f}% >= {self.interrupt_at_percent}%")
            print(f"[INTERRUPT] Sending SIGINT to process {self.process_pid}")
            
            try:
                os.kill(self.process_pid, signal.SIGINT)
                self.interrupted = True
                print("[INTERRUPT] Signal sent successfully")
                return True
            except Exception as e:
                print(f"[INTERRUPT] Failed to send signal: {e}")
                return False
        
        return False


class TimingExtractor:
    """Extract timing information from training logs"""
    
    @staticmethod
    def extract_from_log(log_file: str) -> Dict:
        """Extract all timing information from a log file"""
        timing_data = {
            "rollout_step": [],
            "train_step": [],
            "weights_update_step": [],
            "e2e": [],
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract timing with pattern: "XXX took Y.YY seconds"
                match = re.search(r"(\w+)\s+took\s+([\d.]+)\s+seconds", line)
                if match:
                    phase = match.group(1)
                    time_val = float(match.group(2))
                    if phase in timing_data:
                        timing_data[phase].append(time_val)
        
        # Calculate statistics
        stats = {}
        for phase, times in timing_data.items():
            if times:
                stats[phase] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        
        return stats
    
    @staticmethod
    def extract_checkpoint_info(log_file: str) -> Dict:
        """Extract checkpoint-related information from logs"""
        info = {
            "responses_saved": 0,
            "prompts_completed": 0,
            "checkpoints_loaded": 0,
            "responses_loaded": 0,
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "[ResponseCheckpoint] Saved response" in line:
                    info["responses_saved"] += 1
                elif "[ResponseCheckpoint] Prompt completed" in line:
                    match = re.search(r"completed_prompts=(\d+)", line)
                    if match:
                        info["prompts_completed"] = int(match.group(1))
                elif "[ResponseCheckpoint] Loaded checkpoint" in line:
                    info["checkpoints_loaded"] += 1
                elif "Loading completed response from checkpoint" in line:
                    info["responses_loaded"] += 1
        
        return info


def monitor_and_interrupt(args):
    """Monitor training log and interrupt at specified progress"""
    print(f"Monitoring log file: {args.log_file}")
    print(f"Will interrupt at: {args.interrupt_at}% progress")
    
    if args.pid:
        print(f"Target process PID: {args.pid}")
        interrupter = TrainingInterrupter(args.pid, args.interrupt_at)
    else:
        print("No PID provided, will only monitor (no interruption)")
        interrupter = None
    
    monitor = LogMonitor(args.log_file)
    
    try:
        for line, parsed in monitor.tail_log():
            if parsed:
                if parsed.get("type") == "prompt_completed":
                    progress = monitor.get_progress_percentage()
                    print(f"[PROGRESS] {parsed['completed']}/{parsed['total']} prompts ({progress:.1f}%)")
                    
                    if interrupter:
                        if interrupter.check_and_interrupt(progress):
                            print("[MONITOR] Interrupted. Exiting monitor.")
                            break
                
                elif parsed.get("type") == "response_saved":
                    print(f"[SAVED] Response for query {parsed['query_id']}, idx {parsed['response_idx']}")
                
                elif parsed.get("type") == "checkpoint_loaded":
                    print(f"[LOADED] Checkpoint for step {parsed['step']}, {parsed['completed_prompts']} prompts")
    
    except KeyboardInterrupt:
        print("\n[MONITOR] Interrupted by user")
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {args.log_file}")


def extract_timing(args):
    """Extract timing information from log file"""
    print(f"Extracting timing from: {args.log_file}")
    
    stats = TimingExtractor.extract_from_log(args.log_file)
    checkpoint_info = TimingExtractor.extract_checkpoint_info(args.log_file)
    
    print("\n=== Timing Statistics ===")
    for phase, data in stats.items():
        print(f"\n{phase}:")
        print(f"  Count: {data['count']}")
        print(f"  Total: {data['total']:.2f}s")
        print(f"  Mean:  {data['mean']:.2f}s")
        print(f"  Min:   {data['min']:.2f}s")
        print(f"  Max:   {data['max']:.2f}s")
    
    print("\n=== Checkpoint Information ===")
    for key, value in checkpoint_info.items():
        print(f"  {key}: {value}")
    
    # Save to JSON
    output = {
        "timing_stats": stats,
        "checkpoint_info": checkpoint_info,
        "extracted_at": datetime.now().isoformat(),
        "log_file": args.log_file,
    }
    
    output_file = args.output or "timing_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


def create_test_dataset(args):
    """Create test dataset from source file"""
    print(f"Creating test dataset:")
    print(f"  Source: {args.source}")
    print(f"  Output: {args.output}")
    print(f"  Size: {args.size}")
    
    if not os.path.exists(args.source):
        print(f"ERROR: Source file not found: {args.source}")
        return
    
    samples = []
    with open(args.source, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.size:
                break
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                print(f"WARNING: Skipping invalid JSON at line {i+1}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Created dataset with {len(samples)} samples")


def generate_test_report(args):
    """Generate a comprehensive test report"""
    print("Generating test report...")
    
    # Load all result files
    results = []
    for result_file in args.result_files:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results.append(json.load(f))
    
    # Generate markdown report
    report_lines = [
        "# Response Checkpoint Test Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Summary",
        "",
    ]
    
    # Add results
    for i, result in enumerate(results):
        report_lines.append(f"### Test {i+1}")
        report_lines.append("```json")
        report_lines.append(json.dumps(result, indent=2))
        report_lines.append("```")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    output_file = args.output or "test_report.md"
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Helper utilities for response checkpoint testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training and interrupt at progress")
    monitor_parser.add_argument("--log-file", required=True, help="Path to training log file")
    monitor_parser.add_argument("--interrupt-at", type=float, default=50.0, help="Interrupt at this progress percentage")
    monitor_parser.add_argument("--pid", type=int, help="Process ID to interrupt")
    
    # Extract timing command
    timing_parser = subparsers.add_parser("extract-timing", help="Extract timing from logs")
    timing_parser.add_argument("--log-file", required=True, help="Path to training log file")
    timing_parser.add_argument("--output", help="Output JSON file")
    
    # Create dataset command
    dataset_parser = subparsers.add_parser("create-dataset", help="Create test dataset")
    dataset_parser.add_argument("--source", required=True, help="Source JSONL file")
    dataset_parser.add_argument("--output", required=True, help="Output JSONL file")
    dataset_parser.add_argument("--size", type=int, required=True, help="Number of samples")
    
    # Generate report command
    report_parser = subparsers.add_parser("generate-report", help="Generate test report")
    report_parser.add_argument("--result-files", nargs="+", required=True, help="Result JSON files")
    report_parser.add_argument("--output", help="Output markdown file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "monitor":
        monitor_and_interrupt(args)
    elif args.command == "extract-timing":
        extract_timing(args)
    elif args.command == "create-dataset":
        create_test_dataset(args)
    elif args.command == "generate-report":
        generate_test_report(args)


if __name__ == "__main__":
    main()

