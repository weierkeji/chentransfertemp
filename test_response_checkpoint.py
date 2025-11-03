"""
Response Checkpoint Testing Script

This script provides comprehensive testing for the response_checkpoint functionality,
including fault tolerance tests and performance measurements.

Usage:
    # Run all tests
    python tests/test_response_checkpoint.py --all
    
    # Run only fault tolerance tests
    python tests/test_response_checkpoint.py --fault-tolerance
    
    # Run only performance tests
    python tests/test_response_checkpoint.py --performance
    
    # Verify checkpoint files
    python tests/test_response_checkpoint.py --verify-checkpoint <checkpoint_path>
"""

import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


class CheckpointVerifier:
    """Utility class to verify and inspect checkpoint files"""
    
    @staticmethod
    def verify_checkpoint_files(checkpoint_dir: str) -> Tuple[bool, str]:
        """
        Verify that checkpoint files exist and are valid
        
        Returns:
            (success, message)
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return False, f"Checkpoint directory does not exist: {checkpoint_dir}"
        
        meta_file = checkpoint_path / "response_meta.pkl"
        ckpt_file = checkpoint_path / "response_ckpt.pkl"
        
        if not meta_file.exists():
            return False, f"Missing response_meta.pkl in {checkpoint_dir}"
        
        if not ckpt_file.exists():
            return False, f"Missing response_ckpt.pkl in {checkpoint_dir}"
        
        # Try to load files
        try:
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            with open(ckpt_file, "rb") as f:
                checkpoint = pickle.load(f)
            
            return True, f"Checkpoint valid: {len(checkpoint)} prompts, {meta.completed_prompts} completed"
        except Exception as e:
            return False, f"Failed to load checkpoint: {str(e)}"
    
    @staticmethod
    def inspect_checkpoint(checkpoint_dir: str):
        """Print detailed information about a checkpoint"""
        checkpoint_path = Path(checkpoint_dir)
        meta_file = checkpoint_path / "response_meta.pkl"
        ckpt_file = checkpoint_path / "response_ckpt.pkl"
        
        if not meta_file.exists() or not ckpt_file.exists():
            print_error(f"Checkpoint files not found in {checkpoint_dir}")
            return
        
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        with open(ckpt_file, "rb") as f:
            checkpoint = pickle.load(f)
        
        print_header("Checkpoint Inspection")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        print(f"\nMetadata:")
        print(f"  Global Step: {meta.global_step}")
        print(f"  Epoch: {meta.epoch}")
        print(f"  Epoch Step: {meta.epoch_step}")
        print(f"  Model Version: {meta.model_version}")
        print(f"  Checkpoint Time: {meta.checkpoint_time}")
        print(f"  Total Prompts: {meta.total_prompts}")
        print(f"  Completed Prompts: {meta.completed_prompts}")
        print(f"  N Samples Per Prompt: {meta.n_samples_per_prompt}")
        print(f"  Progress: {meta.completed_prompts}/{meta.total_prompts} ({meta.completed_prompts/max(meta.total_prompts, 1)*100:.1f}%)")
        
        print(f"\nCheckpoint Data:")
        print(f"  Total Query IDs: {len(checkpoint)}")
        
        total_responses = 0
        for query_id, data in checkpoint.items():
            completed_samples = data.get("completed_samples", 0)
            n_samples = data.get("n_samples", 0)
            total_responses += completed_samples
            print(f"  Query {query_id}: {completed_samples}/{n_samples} responses")
        
        print(f"\n  Total Responses Saved: {total_responses}")
        
        # File sizes
        meta_size = meta_file.stat().st_size / 1024  # KB
        ckpt_size = ckpt_file.stat().st_size / (1024 * 1024)  # MB
        print(f"\nFile Sizes:")
        print(f"  response_meta.pkl: {meta_size:.2f} KB")
        print(f"  response_ckpt.pkl: {ckpt_size:.2f} MB")


class TestDatasetGenerator:
    """Generate test datasets from existing training data"""
    
    @staticmethod
    def create_test_dataset(source_path: str, output_path: str, num_samples: int):
        """
        Create a test dataset by sampling from a larger dataset
        
        Args:
            source_path: Path to source JSONL file
            output_path: Path to output test dataset
            num_samples: Number of samples to extract
        """
        print_info(f"Creating test dataset: {output_path}")
        print_info(f"  Source: {source_path}")
        print_info(f"  Samples: {num_samples}")
        
        if not os.path.exists(source_path):
            print_error(f"Source dataset not found: {source_path}")
            return False
        
        try:
            samples = []
            with open(source_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    samples.append(json.loads(line))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print_success(f"Created test dataset with {len(samples)} samples")
            return True
        except Exception as e:
            print_error(f"Failed to create test dataset: {str(e)}")
            return False


class TrainingRunner:
    """Utility to run training and monitor progress"""
    
    def __init__(self, config_path: str, python_path: str = "python"):
        self.config_path = config_path
        self.python_path = python_path
        self.process = None
        self.log_file = None
    
    def start_training(self, trainer_script: str, log_path: Optional[str] = None):
        """
        Start training process
        
        Args:
            trainer_script: Path to trainer script
            log_path: Optional path to log file
        """
        if log_path:
            self.log_file = open(log_path, 'w')
        else:
            self.log_file = subprocess.PIPE
        
        cmd = [
            self.python_path,
            trainer_script,
            f"--config={self.config_path}"
        ]
        
        print_info(f"Starting training: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        return self.process
    
    def wait_for_condition(self, condition_fn, timeout: float = 600, check_interval: float = 5):
        """
        Wait for a condition to be met by checking logs
        
        Args:
            condition_fn: Function that takes log content and returns True if condition is met
            timeout: Maximum time to wait in seconds
            check_interval: How often to check in seconds
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                print_warning("Training process exited")
                return False
            
            time.sleep(check_interval)
            
            # Check condition (would need to read from log file)
            # This is a simplified implementation
            if condition_fn():
                return True
        
        print_warning("Timeout waiting for condition")
        return False
    
    def interrupt_training(self):
        """Send interrupt signal to training process"""
        if self.process and self.process.poll() is None:
            print_info("Sending interrupt signal to training process")
            self.process.send_signal(signal.SIGINT)
            time.sleep(5)  # Give it time to cleanup
            
            if self.process.poll() is None:
                print_warning("Process still running, force killing")
                self.process.kill()
            
            self.process.wait()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.log_file and self.log_file != subprocess.PIPE:
            self.log_file.close()


class FaultToleranceTests:
    """Test suite for fault tolerance"""
    
    def __init__(self, config_path: str, checkpoint_root: str):
        self.config_path = config_path
        self.checkpoint_root = checkpoint_root
        self.verifier = CheckpointVerifier()
    
    def test_scenario_a_basic_recovery(self):
        """
        Test Scenario A: Basic checkpoint save and recovery
        
        Steps:
        1. Start training with response_checkpoint enabled
        2. Interrupt after ~50% completion
        3. Verify checkpoint files exist and are valid
        4. Restart training
        5. Verify recovery and completion
        """
        print_header("Test Scenario A: Basic Recovery")
        
        print_info("Step 1: Starting training with response_checkpoint enabled")
        print_warning("  Manual step: Start training and monitor progress")
        print_warning("  Interrupt training when ~50% prompts are completed (Ctrl+C)")
        
        input("\nPress Enter when you have interrupted the training...")
        
        print_info("\nStep 2: Verifying checkpoint files")
        
        # Find checkpoint directory for step 0
        checkpoint_dir = os.path.join(self.checkpoint_root, "step_0_even")
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = os.path.join(self.checkpoint_root, "step_0_odd")
        
        success, message = self.verifier.verify_checkpoint_files(checkpoint_dir)
        if success:
            print_success(message)
        else:
            print_error(message)
            return False
        
        # Inspect checkpoint
        self.verifier.inspect_checkpoint(checkpoint_dir)
        
        print_info("\nStep 3: Restart training to test recovery")
        print_warning("  Manual step: Restart training with the same configuration")
        print_warning("  Observe logs for checkpoint recovery messages")
        print_warning("  Verify that already-completed responses are skipped")
        
        input("\nPress Enter when training has completed...")
        
        print_success("Scenario A completed. Please manually verify:")
        print("  ✓ Training completed successfully")
        print("  ✓ Logs show checkpoint was loaded")
        print("  ✓ Logs show skipping of completed responses")
        print("  ✓ Final training data metrics are correct")
        
        return True
    
    def test_scenario_b_multiple_interruptions(self):
        """
        Test Scenario B: Multiple interruptions and recoveries
        """
        print_header("Test Scenario B: Multiple Interruptions")
        
        print_info("This test requires multiple manual interruptions:")
        print("  1. Interrupt at ~25% completion")
        print("  2. Restart, then interrupt at ~60% completion")
        print("  3. Restart and let complete")
        
        print_warning("\nThis is a manual test. Please follow the steps in the test plan.")
        
        return True
    
    def test_scenario_c_error_handling(self):
        """
        Test Scenario C: Error handling and edge cases
        """
        print_header("Test Scenario C: Error Handling")
        
        print_info("Testing checkpoint corruption handling")
        
        checkpoint_dir = os.path.join(self.checkpoint_root, "step_0_even")
        if not os.path.exists(checkpoint_dir):
            print_warning(f"Checkpoint directory not found: {checkpoint_dir}")
            print_info("Create a checkpoint first by running and interrupting training")
            return False
        
        # Backup original checkpoint
        backup_dir = checkpoint_dir + "_backup"
        import shutil
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(checkpoint_dir, backup_dir)
        print_success(f"Backed up checkpoint to {backup_dir}")
        
        # Test 1: Corrupt checkpoint file
        print_info("\nTest 1: Corrupting checkpoint file")
        ckpt_file = os.path.join(checkpoint_dir, "response_ckpt.pkl")
        with open(ckpt_file, 'wb') as f:
            f.write(b"corrupted data")
        
        print_warning("Checkpoint corrupted. Restart training to test recovery.")
        print_info("Expected: Training should detect corruption and start fresh")
        
        input("\nPress Enter after testing corrupted checkpoint...")
        
        # Restore checkpoint
        shutil.rmtree(checkpoint_dir)
        shutil.copytree(backup_dir, checkpoint_dir)
        print_success("Checkpoint restored")
        
        return True


class PerformanceTests:
    """Test suite for performance measurements"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.results = []
    
    def measure_baseline(self, n_samples: int) -> Dict:
        """
        Measure baseline performance without interruption
        
        Returns:
            Dictionary with timing results
        """
        print_header(f"Baseline Performance Test (n_samples={n_samples})")
        
        print_info("Measuring complete rollout time without interruption")
        print_warning("Manual step: Run training and record rollout_step timing from logs")
        
        result = {
            "test_type": "baseline",
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
            "rollout_time": None,
        }
        
        rollout_time = input("\nEnter rollout_step time from logs (in seconds): ")
        try:
            result["rollout_time"] = float(rollout_time)
            print_success(f"Baseline rollout time: {result['rollout_time']:.2f}s")
        except ValueError:
            print_error("Invalid time value")
            return result
        
        self.results.append(result)
        return result
    
    def measure_with_interruption(
        self, 
        n_samples: int, 
        interruption_point: str,
        enable_checkpoint: bool
    ) -> Dict:
        """
        Measure performance with interruption
        
        Args:
            n_samples: Number of samples per prompt
            interruption_point: "25%", "50%", or "75%"
            enable_checkpoint: Whether response_checkpoint is enabled
        """
        test_type = "with_checkpoint" if enable_checkpoint else "without_checkpoint"
        print_header(f"Interruption Test: {interruption_point} - {test_type} (n_samples={n_samples})")
        
        print_info(f"Test configuration:")
        print(f"  Interruption point: {interruption_point}")
        print(f"  Response checkpoint: {'Enabled' if enable_checkpoint else 'Disabled'}")
        print(f"  N samples: {n_samples}")
        
        result = {
            "test_type": test_type,
            "n_samples": n_samples,
            "interruption_point": interruption_point,
            "timestamp": datetime.now().isoformat(),
            "time_before_interrupt": None,
            "time_after_restart": None,
            "total_time": None,
        }
        
        print_warning(f"\nManual steps:")
        print(f"  1. Start training (checkpoint {'enabled' if enable_checkpoint else 'disabled'})")
        print(f"  2. Interrupt at {interruption_point} completion")
        print(f"  3. Record time before interrupt")
        print(f"  4. Restart training")
        print(f"  5. Record time after restart")
        
        time_before = input("\nEnter rollout time before interrupt (seconds): ")
        time_after = input("Enter rollout time after restart (seconds): ")
        
        try:
            result["time_before_interrupt"] = float(time_before)
            result["time_after_restart"] = float(time_after)
            result["total_time"] = result["time_before_interrupt"] + result["time_after_restart"]
            
            print_success(f"Time before interrupt: {result['time_before_interrupt']:.2f}s")
            print_success(f"Time after restart: {result['time_after_restart']:.2f}s")
            print_success(f"Total time: {result['total_time']:.2f}s")
        except ValueError:
            print_error("Invalid time values")
            return result
        
        self.results.append(result)
        return result
    
    def compare_results(self):
        """Compare and display performance results"""
        print_header("Performance Test Results Summary")
        
        if not self.results:
            print_warning("No results to compare")
            return
        
        # Group results by interruption point and n_samples
        grouped = {}
        for result in self.results:
            if result["test_type"] == "baseline":
                continue
            
            key = (result["n_samples"], result["interruption_point"])
            if key not in grouped:
                grouped[key] = {"with": None, "without": None}
            
            if result["test_type"] == "with_checkpoint":
                grouped[key]["with"] = result
            else:
                grouped[key]["without"] = result
        
        # Compare results
        for (n_samples, interrupt_point), data in grouped.items():
            print(f"\n{Colors.BOLD}Configuration: n_samples={n_samples}, interruption={interrupt_point}{Colors.ENDC}")
            
            if data["without"] and data["with"]:
                time_without = data["without"]["total_time"]
                time_with = data["with"]["total_time"]
                
                if time_without and time_with:
                    savings = time_without - time_with
                    savings_pct = (savings / time_without) * 100
                    
                    print(f"  Without checkpoint: {time_without:.2f}s")
                    print(f"  With checkpoint:    {time_with:.2f}s")
                    print(f"  Time saved:         {savings:.2f}s ({savings_pct:.1f}%)")
                    
                    if savings_pct > 0:
                        print_success(f"  ✓ Checkpoint saves {savings_pct:.1f}% time")
                    else:
                        print_warning(f"  ⚠ Checkpoint overhead: {-savings_pct:.1f}%")
            else:
                print_warning("  Incomplete data for comparison")
        
        # Save results to file
        results_file = "response_checkpoint_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print_success(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test response_checkpoint functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--fault-tolerance",
        action="store_true",
        help="Run fault tolerance tests"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests"
    )
    parser.add_argument(
        "--verify-checkpoint",
        type=str,
        metavar="PATH",
        help="Verify and inspect a checkpoint directory"
    )
    parser.add_argument(
        "--create-dataset",
        nargs=3,
        metavar=("SOURCE", "OUTPUT", "NUM_SAMPLES"),
        help="Create test dataset from source file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/test_response_checkpoint.yaml",
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="/storage/openpsi/experiments/recover/",
        help="Root directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Handle verify checkpoint
    if args.verify_checkpoint:
        verifier = CheckpointVerifier()
        verifier.inspect_checkpoint(args.verify_checkpoint)
        return
    
    # Handle create dataset
    if args.create_dataset:
        source, output, num_samples = args.create_dataset
        generator = TestDatasetGenerator()
        generator.create_test_dataset(source, output, int(num_samples))
        return
    
    # Run tests
    if not any([args.all, args.fault_tolerance, args.performance]):
        parser.print_help()
        return
    
    print_header("Response Checkpoint Testing Suite")
    print_info(f"Configuration: {args.config}")
    print_info(f"Checkpoint root: {args.checkpoint_root}")
    
    # Fault tolerance tests
    if args.all or args.fault_tolerance:
        ft_tests = FaultToleranceTests(args.config, args.checkpoint_root)
        
        print("\n")
        ft_tests.test_scenario_a_basic_recovery()
        
        print("\n")
        ft_tests.test_scenario_b_multiple_interruptions()
        
        print("\n")
        ft_tests.test_scenario_c_error_handling()
    
    # Performance tests
    if args.all or args.performance:
        perf_tests = PerformanceTests(args.config)
        
        print("\n")
        print_info("Performance tests require manual timing measurements")
        print_info("Follow the prompts to record timing data")
        
        # Test with different n_samples
        for n_samples in [4, 8]:
            perf_tests.measure_baseline(n_samples)
            
            for interrupt_point in ["50%"]:  # Can add "25%", "75%" for more comprehensive testing
                perf_tests.measure_with_interruption(n_samples, interrupt_point, False)
                perf_tests.measure_with_interruption(n_samples, interrupt_point, True)
        
        perf_tests.compare_results()
    
    print_header("Testing Complete")
    print_success("All tests finished. Review results above.")


if __name__ == "__main__":
    main()

