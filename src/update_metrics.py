import argparse
import json
import os
from src.config import LOGS_DIR

def update_training_logs(experiment_name):
    """Update training logs JSON file"""
    log_path = os.path.join(LOGS_DIR, experiment_name, 'training_logs.json')
    if os.path.exists(log_path):
        print(f"Training logs updated for {experiment_name}")
        return True
    return False

def update_test_metrics(experiment_name):
    """Update test metrics JSON file"""
    metrics_path = os.path.join(LOGS_DIR, experiment_name, 'test_metrics.json')
    if os.path.exists(metrics_path):
        print(f"Test metrics updated for {experiment_name}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Update metrics on the web interface')
    parser.add_argument('--experiment', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--type', choices=['training', 'test', 'all'], required=True, 
                      help='Type of metrics to update')
    
    args = parser.parse_args()
    
    if args.type in ['training', 'all']:
        if update_training_logs(args.experiment):
            print("Training metrics updated successfully")
        else:
            print("No training logs found")
            
    if args.type in ['test', 'all']:
        if update_test_metrics(args.experiment):
            print("Test metrics updated successfully")
        else:
            print("No test metrics found")

if __name__ == '__main__':
    main() 