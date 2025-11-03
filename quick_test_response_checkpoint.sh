#!/bin/bash
# Quick test script for response_checkpoint functionality
# This script automates common testing scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/examples/configs/test_response_checkpoint.yaml"
TEST_DATASET_SOURCE="/storage/dataset/nlp/areal/moe_lite_math_0527_merge_train_areal.jsonl"
TEST_DATASET_SMALL="/storage/dataset/nlp/areal/test_dataset_small.jsonl"
TEST_DATASET_MEDIUM="/storage/dataset/nlp/areal/test_dataset_medium.jsonl"
CHECKPOINT_ROOT="/storage/openpsi/experiments/recover/$USER/test-response-checkpoint/test-trial/response_checkpoints"
LOG_DIR="$PROJECT_ROOT/test_logs"

# Create log directory
mkdir -p "$LOG_DIR"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to create test datasets
create_datasets() {
    print_header "Creating Test Datasets"
    
    if [ ! -f "$TEST_DATASET_SOURCE" ]; then
        print_error "Source dataset not found: $TEST_DATASET_SOURCE"
        print_info "Please update TEST_DATASET_SOURCE in this script"
        exit 1
    fi
    
    print_info "Creating small dataset (50 samples)..."
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" create-dataset \
        --source "$TEST_DATASET_SOURCE" \
        --output "$TEST_DATASET_SMALL" \
        --size 50
    
    print_info "Creating medium dataset (200 samples)..."
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" create-dataset \
        --source "$TEST_DATASET_SOURCE" \
        --output "$TEST_DATASET_MEDIUM" \
        --size 200
    
    print_success "Test datasets created"
}

# Function to verify checkpoint
verify_checkpoint() {
    local step=${1:-0}
    local parity="even"
    if [ $((step % 2)) -eq 1 ]; then
        parity="odd"
    fi
    
    local ckpt_dir="$CHECKPOINT_ROOT/step_${step}_${parity}"
    
    print_header "Verifying Checkpoint"
    print_info "Checkpoint directory: $ckpt_dir"
    
    if [ ! -d "$ckpt_dir" ]; then
        print_warning "Checkpoint directory not found"
        # Try the other parity
        if [ "$parity" = "even" ]; then
            parity="odd"
        else
            parity="even"
        fi
        ckpt_dir="$CHECKPOINT_ROOT/step_${step}_${parity}"
        
        if [ ! -d "$ckpt_dir" ]; then
            print_error "No checkpoint found for step $step"
            return 1
        fi
    fi
    
    python "$SCRIPT_DIR/test_response_checkpoint.py" verify-checkpoint "$ckpt_dir"
}

# Function to run basic fault tolerance test
test_fault_tolerance() {
    print_header "Fault Tolerance Test"
    
    local log_file="$LOG_DIR/fault_tolerance_test.log"
    local interrupt_at=${1:-50}
    
    print_info "Step 1: Starting training with checkpoint enabled"
    print_info "Log file: $log_file"
    
    # Start training in background
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$log_file" 2>&1 &
    
    local train_pid=$!
    print_success "Training started (PID: $train_pid)"
    
    print_info "Step 2: Monitoring progress (will interrupt at ${interrupt_at}%)"
    
    # Monitor and interrupt
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" monitor \
        --log-file "$log_file" \
        --interrupt-at "$interrupt_at" \
        --pid "$train_pid" || true
    
    print_success "Training interrupted"
    
    print_info "Step 3: Verifying checkpoint was created"
    sleep 2
    verify_checkpoint 0
    
    print_info "Step 4: Restarting training to test recovery"
    local restart_log="$LOG_DIR/fault_tolerance_restart.log"
    
    print_warning "Press Ctrl+C after a few responses are loaded from checkpoint"
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$restart_log" 2>&1 || true
    
    print_info "Step 5: Checking restart logs"
    if grep -q "Loaded checkpoint" "$restart_log"; then
        print_success "Checkpoint was loaded on restart"
    else
        print_error "Checkpoint was NOT loaded on restart"
    fi
    
    if grep -q "Loading completed response" "$restart_log"; then
        print_success "Responses were loaded from checkpoint"
    else
        print_warning "No responses loaded from checkpoint (check logs)"
    fi
    
    print_success "Fault tolerance test completed"
    print_info "Review logs:"
    print_info "  Initial run: $log_file"
    print_info "  Restart run: $restart_log"
}

# Function to run performance comparison test
test_performance() {
    print_header "Performance Comparison Test"
    
    local interrupt_at=${1:-50}
    
    print_info "This test will compare performance with and without checkpoint"
    print_info "Interruption point: ${interrupt_at}%"
    
    # Test without checkpoint
    print_info "=== Test 1: WITHOUT checkpoint ==="
    
    local log1="$LOG_DIR/perf_no_ckpt_run1.log"
    local log2="$LOG_DIR/perf_no_ckpt_run2.log"
    
    print_info "Temporarily disabling checkpoint in config..."
    # Note: This requires sed or manual config change
    print_warning "Please edit $CONFIG_FILE and set enable_response_checkpoint: false"
    read -p "Press Enter when ready..."
    
    print_info "Run 1: Interrupting at ${interrupt_at}%"
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$log1" 2>&1 &
    local pid1=$!
    
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" monitor \
        --log-file "$log1" \
        --interrupt-at "$interrupt_at" \
        --pid "$pid1" || true
    
    print_info "Run 2: Starting fresh"
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$log2" 2>&1
    
    print_success "Test without checkpoint completed"
    
    # Test with checkpoint
    print_info "=== Test 2: WITH checkpoint ==="
    
    local log3="$LOG_DIR/perf_with_ckpt_run1.log"
    local log4="$LOG_DIR/perf_with_ckpt_run2.log"
    
    print_info "Enabling checkpoint in config..."
    print_warning "Please edit $CONFIG_FILE and set enable_response_checkpoint: true"
    read -p "Press Enter when ready..."
    
    # Clean up old checkpoints
    print_info "Cleaning up old checkpoints..."
    rm -rf "$CHECKPOINT_ROOT"
    
    print_info "Run 1: Interrupting at ${interrupt_at}%"
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$log3" 2>&1 &
    local pid2=$!
    
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" monitor \
        --log-file "$log3" \
        --interrupt-at "$interrupt_at" \
        --pid "$pid2" || true
    
    print_info "Run 2: Recovering from checkpoint"
    python "$PROJECT_ROOT/arobust/grpo_trainer.py" \
        --config="$CONFIG_FILE" > "$log4" 2>&1
    
    print_success "Test with checkpoint completed"
    
    # Extract timing
    print_info "Extracting timing data..."
    
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" extract-timing \
        --log-file "$log1" --output "$LOG_DIR/no_ckpt_run1.json"
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" extract-timing \
        --log-file "$log2" --output "$LOG_DIR/no_ckpt_run2.json"
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" extract-timing \
        --log-file "$log3" --output "$LOG_DIR/with_ckpt_run1.json"
    python "$SCRIPT_DIR/test_response_checkpoint_helper.py" extract-timing \
        --log-file "$log4" --output "$LOG_DIR/with_ckpt_run2.json"
    
    print_success "Performance test completed"
    print_info "Results saved in: $LOG_DIR"
    print_info "Use Python script to analyze and compare timing data"
}

# Function to clean up test artifacts
cleanup() {
    print_header "Cleaning Up Test Artifacts"
    
    print_info "Removing test logs..."
    rm -rf "$LOG_DIR"
    
    print_info "Removing checkpoints..."
    rm -rf "$CHECKPOINT_ROOT"
    
    print_info "Removing test datasets..."
    rm -f "$TEST_DATASET_SMALL" "$TEST_DATASET_MEDIUM"
    
    print_success "Cleanup completed"
}

# Main menu
show_menu() {
    clear
    print_header "Response Checkpoint Quick Test"
    echo ""
    echo "1) Create test datasets"
    echo "2) Run fault tolerance test"
    echo "3) Run performance comparison test"
    echo "4) Verify checkpoint"
    echo "5) View test logs"
    echo "6) Cleanup test artifacts"
    echo "7) Exit"
    echo ""
}

# Main script
main() {
    if [ $# -gt 0 ]; then
        case "$1" in
            datasets)
                create_datasets
                ;;
            fault-tolerance)
                test_fault_tolerance "${2:-50}"
                ;;
            performance)
                test_performance "${2:-50}"
                ;;
            verify)
                verify_checkpoint "${2:-0}"
                ;;
            cleanup)
                cleanup
                ;;
            *)
                echo "Usage: $0 {datasets|fault-tolerance|performance|verify|cleanup} [options]"
                exit 1
                ;;
        esac
    else
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            
            case $choice in
                1)
                    create_datasets
                    read -p "Press Enter to continue..."
                    ;;
                2)
                    read -p "Interrupt at percentage (default 50): " interrupt
                    interrupt=${interrupt:-50}
                    test_fault_tolerance "$interrupt"
                    read -p "Press Enter to continue..."
                    ;;
                3)
                    read -p "Interrupt at percentage (default 50): " interrupt
                    interrupt=${interrupt:-50}
                    test_performance "$interrupt"
                    read -p "Press Enter to continue..."
                    ;;
                4)
                    read -p "Step number (default 0): " step
                    step=${step:-0}
                    verify_checkpoint "$step"
                    read -p "Press Enter to continue..."
                    ;;
                5)
                    if [ -d "$LOG_DIR" ]; then
                        ls -lh "$LOG_DIR"
                        read -p "View which log file? " logfile
                        if [ -f "$LOG_DIR/$logfile" ]; then
                            less "$LOG_DIR/$logfile"
                        fi
                    else
                        print_warning "No logs found"
                    fi
                    read -p "Press Enter to continue..."
                    ;;
                6)
                    read -p "Are you sure you want to cleanup? (y/N) " confirm
                    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                        cleanup
                    fi
                    read -p "Press Enter to continue..."
                    ;;
                7)
                    print_info "Exiting..."
                    exit 0
                    ;;
                *)
                    print_error "Invalid option"
                    read -p "Press Enter to continue..."
                    ;;
            esac
        done
    fi
}

# Run main
main "$@"

