from flask import Flask, render_template, jsonify, send_file, request
import os
import json
from src.config.base_config import (
    WEBAPP, TRAINING, MODEL, AUGMENTATION, DATASET, LOGS_DIR, 
    EXPERIMENTS, BENCHMARK_EXPERIMENTS
)
from src.models.get_model import get_model, count_parameters
from src.config.hyperparameter_config import (
    BASE_MODEL_ID, 
    LEARNING_RATE_EXPERIMENTS, 
    OPTIMIZER_EXPERIMENTS, 
    SCHEDULER_EXPERIMENTS,
    AUGMENTATION_EXPERIMENTS
)

app = Flask(__name__)

def format_number(value):
    """Format large numbers with M for millions, K for thousands"""
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    return str(value)

def format_learning_rate(lr):
    """Format learning rate in scientific notation (e.g., 1e-3)"""
    return f"{lr:.0e}".replace("e-0", "e-")

app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_lr'] = format_learning_rate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/unet-experiments')
def unet_experiments():
    experiments = {}
    
    # Common configurations that apply to all experiments
    common_config = {
        # Model configurations
        'model': {
            # Remove architecture since it's experiment-specific now
            'in_channels': MODEL['IN_CHANNELS'],
            'num_classes': DATASET['NUM_CLASSES'],
            'class_labels': DATASET['CLASS_LABELS']
        },
        # Training configurations
        'training': {
            'loss_function': TRAINING['LOSS_FUNCTION'],
            'optimizer': TRAINING['OPTIMIZER'],
            'learning_rate': TRAINING['LEARNING_RATE'],
            'batch_size': TRAINING['BATCH_SIZE'],
            'epochs': TRAINING['NUM_EPOCHS']
        },
        # Dataset configurations
        'dataset': {
            'image_size': DATASET['IMAGE_SIZE'],
            'transformations': [
                f"Resize({DATASET['IMAGE_SIZE']}, {DATASET['IMAGE_SIZE']})",
            ]
        }
    }
    
    # Add augmentations to common config
    if AUGMENTATION['TRAIN']['NORMALIZE']:
        common_config['dataset']['transformations'].append('Normalize')
    if AUGMENTATION['TRAIN'].get('HORIZONTAL_FLIP'):
        common_config['dataset']['transformations'].append('RandomHorizontalFlip(p=0.5)')
    if AUGMENTATION['TRAIN'].get('VERTICAL_FLIP'):
        common_config['dataset']['transformations'].append('RandomVerticalFlip(p=0.5)')
    if AUGMENTATION['TRAIN'].get('COLOR_JITTER'):
        common_config['dataset']['transformations'].append(
            f"ColorJitter(brightness={AUGMENTATION['TRAIN']['BRIGHTNESS']}, "
            f"contrast={AUGMENTATION['TRAIN']['CONTRAST']})"
        )

    # Process each experiment
    for exp_id, exp_config in EXPERIMENTS.items():
        # Initialize model for this experiment
        model = get_model(
            architecture=exp_config['architecture'],
            encoder=exp_config['encoder']
        )
        
        # Count parameters
        params = count_parameters(model)
        total_params = params['total']
        trainable_params = params['trainable'] 
        encoder_params = params['encoder']
        decoder_params = params['decoder']
        
        # Store experiment data
        experiments[exp_id] = {
            'name': exp_config['name'],
            'config': exp_config,
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'encoder': encoder_params,
                'decoder': decoder_params
            }
        }

    return render_template('unet_experiments.html', 
                         experiments=experiments,
                         common_config=common_config)

@app.route('/get_training_logs/<experiment_name>')
def get_training_logs(experiment_name):
    log_path = os.path.join(LOGS_DIR, experiment_name, 'training_logs.json')
    print(f"Fetching training logs from: {log_path}")
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                raw_logs = json.load(f)
                
                total_time_seconds = sum(raw_logs['epoch_times'])
                total_time_minutes = total_time_seconds / 60  # Convert to minutes
                
                # Transform the logs to match our frontend's expected format
                logs = {
                    'train_losses': [round(loss, 2) for loss in raw_logs['train_losses']],
                    'val_losses': [round(loss, 2) for loss in raw_logs['val_losses']],
                    'current_epoch': raw_logs['current_epoch'],
                    'avg_epoch_time': round(raw_logs['avg_epoch_time'], 2),
                    'total_train_time': round(total_time_minutes, 2),  # In minutes now
                    'best_train_loss': round(min(raw_logs['train_losses']), 2) if raw_logs['train_losses'] else None,
                    'best_val_loss': round(min(raw_logs['val_losses']), 2) if raw_logs['val_losses'] else None,
                    # Add training and validation metrics
                    'train_metrics': {
                        'IoU_all': [round(iou, 2) for iou in raw_logs['train_metrics']['IoU_all']]
                    },
                    'val_metrics': {
                        'IoU_all': [round(iou, 2) for iou in raw_logs['train_metrics']['IoU_all']]
                    }
                }
                
                print(f"Transformed training logs: {logs}")
                return jsonify(logs)
        except Exception as e:
            print(f"Error loading logs: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        print(f"Log file not found at: {log_path}")
        return jsonify({
            'train_losses': [],
            'val_losses': [],
            'best_train_loss': None,
            'best_val_loss': None,
            'avg_epoch_time': None,
            'total_train_time': None,
            'current_epoch': None,
            'train_metrics': {'IoU_all': []},
            'val_metrics': {'IoU_all': []}
        })

@app.route('/get_test_metrics/<experiment_name>')
def get_test_metrics(experiment_name):
    metrics_path = os.path.join(LOGS_DIR, experiment_name, 'test_metrics.json')
    print(f"Looking for test metrics at: {metrics_path}")
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                raw_metrics = json.load(f)
                
                # Helper function to round values in a nested dict
                def round_values(d):
                    return {k: round(v, 2) for k, v in d.items()}
                
                # Transform the metrics to match our frontend's expected format
                metrics = {
                    'iou': {
                        'all': round(raw_metrics['IoU_all'], 2),
                        'building': round(raw_metrics['IoU_Building'], 2),
                        'boundary': round(raw_metrics['IoU_Boundary'], 2),
                        'contact': round(raw_metrics['IoU_Contact Point'], 2)
                    },
                    'f1-score': {
                        'all': round(raw_metrics['F1_Score_all'], 2),
                        'building': round(raw_metrics['F1_Score_Building'], 2),
                        'boundary': round(raw_metrics['F1_Score_Boundary'], 2),
                        'contact': round(raw_metrics['F1_Score_Contact Point'], 2)
                    },
                    'accuracy': {
                        'all': round(raw_metrics['Accuracy_all'], 2),
                        'building': round(raw_metrics['Accuracy_Building'], 2),
                        'boundary': round(raw_metrics['Accuracy_Boundary'], 2),
                        'contact': round(raw_metrics['Accuracy_Contact Point'], 2)
                    },
                    'precision': {
                        'all': round(raw_metrics['Precision_all'], 2),
                        'building': round(raw_metrics['Precision_Building'], 2),
                        'boundary': round(raw_metrics['Precision_Boundary'], 2),
                        'contact': round(raw_metrics['Precision_Contact Point'], 2)
                    },
                    'recall': {
                        'all': round(raw_metrics['Recall_all'], 2),
                        'building': round(raw_metrics['Recall_Building'], 2),
                        'boundary': round(raw_metrics['Recall_Boundary'], 2),
                        'contact': round(raw_metrics['Recall_Contact Point'], 2)
                    }
                }
                print(f"Transformed metrics: {metrics}")
                return jsonify(metrics)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        print(f"Metrics file not found at: {metrics_path}")
        return jsonify({
            'iou': {'all': None, 'building': None, 'boundary': None, 'contact': None},
            'f1-score': {'all': None, 'building': None, 'boundary': None, 'contact': None},
            'accuracy': {'all': None, 'building': None, 'boundary': None, 'contact': None},
            'precision': {'all': None, 'building': None, 'boundary': None, 'contact': None},
            'recall': {'all': None, 'building': None, 'boundary': None, 'contact': None}
        })

@app.route('/get_all_experiments_data')
def get_all_experiments_data():
    all_data = {}
    
    for exp_id, exp_config in EXPERIMENTS.items():
        try:
            # Initialize model to get parameters
            model = get_model(
                architecture=exp_config['architecture'],
                encoder=exp_config['encoder']
            )
            
            # Count parameters
            params = count_parameters(model)
            
            exp_data = {
                'name': exp_config['name'],
                'model_parameters': {
                    'total': int(params['total']),  # Convert to int for JSON
                    'trainable': int(params['trainable']),
                    'encoder': int(params['encoder']),
                    'decoder': int(params['decoder'])
                },
                'training_time': None,
                'train_losses': [],
                'test_metrics': None
            }
            
            # Get training logs
            log_path = os.path.join(LOGS_DIR, exp_id, 'training_logs.json')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
                    exp_data['train_losses'] = logs['train_losses']
                    total_time_minutes = sum(logs['epoch_times']) / 60
                    exp_data['training_time'] = round(total_time_minutes, 2)
            
            # Get test metrics
            metrics_path = os.path.join(LOGS_DIR, exp_id, 'test_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    exp_data['test_metrics'] = json.load(f)
            
            all_data[exp_id] = exp_data
            
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {str(e)}")
            continue
    
    print("Sending experiments data:", all_data)
    return jsonify(all_data)

@app.route('/hyperparameter-tuning')
def hyperparameter_tuning():
    try:
        # Get base model config
        base_config = EXPERIMENTS[BASE_MODEL_ID]
        
        # Initialize model to get parameters
        model = get_model(
            architecture=base_config['architecture'],
            encoder=base_config['encoder']
        )
        
        # Count parameters
        params = count_parameters(model)
        
        base_model_data = {
            'name': base_config['name'],
            'encoder': base_config['encoder'],
            'parameters': {
                'total': int(params['total']),
                'trainable': int(params['trainable']),
                'encoder': int(params['encoder']),
                'decoder': int(params['decoder'])
            },
            'hyperparameters': {
                'learning_rate': TRAINING['LEARNING_RATE'],
                'optimizer': TRAINING['OPTIMIZER'],
                'scheduler': TRAINING.get('SCHEDULER', 'None'),
                'batch_size': TRAINING['BATCH_SIZE']
            },
            'metrics': {
                'iou_building': 0,
                'f1_building': 0,
                'training_time': 0
            }
        }

        # Get training logs from base model directory
        log_path = os.path.join(LOGS_DIR, BASE_MODEL_ID, 'training_logs.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
                total_time_minutes = sum(logs['epoch_times']) / 60
                base_model_data['metrics']['training_time'] = round(total_time_minutes, 2)
        
        # Get test metrics from base model directory
        metrics_path = os.path.join(LOGS_DIR, BASE_MODEL_ID, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                base_metrics = json.load(f)
                # Update specific metrics we want to display
                base_model_data['metrics']['iou_building'] = base_metrics.get('IoU_Building', 0)
                base_model_data['metrics']['f1_building'] = base_metrics.get('F1_Score_Building', 0)
        
        return render_template('hyperparameter_tuning.html', 
                             base_model=base_model_data,
                             base_metrics_json=json.dumps(base_metrics))
        
    except Exception as e:
        print(f"Error processing base model data: {str(e)}")
        return render_template('hyperparameter_tuning.html', base_model={
            'name': 'Error loading model',
            'encoder': 'Unknown',
            'parameters': {'total': 0},
            'hyperparameters': {
                'learning_rate': 0,
                'optimizer': 'Unknown',
                'scheduler': 'Unknown',
                'batch_size': 0
            },
            'metrics': {
                'iou_building': 0,
                'f1_building': 0,
                'training_time': 0
            }
        })

@app.route('/get_hyperparameter_experiments')
def get_hyperparameter_experiments():
    experiment_type = request.args.get('type', 'lr')
    all_data = {}
    
    base_model_dir = os.path.join(LOGS_DIR, BASE_MODEL_ID)
    
    # Add base model data first
    base_log_path = os.path.join(LOGS_DIR, BASE_MODEL_ID, 'training_logs.json')
    if os.path.exists(base_log_path):
        with open(base_log_path, 'r') as f:
            base_logs = json.load(f)
            all_data['base_model'] = {
                'name': 'Base Model',
                'training_time': None,
                'train_losses': base_logs.get('train_losses', []),
                'val_losses': base_logs.get('val_losses', []),
                'train_metrics': base_logs.get('train_metrics', {}),
                'val_metrics': base_logs.get('val_metrics', {}),
                'test_metrics': None
            }
            
            # Get base model test metrics
            base_metrics_path = os.path.join(LOGS_DIR, BASE_MODEL_ID, 'test_metrics.json')
            if os.path.exists(base_metrics_path):
                with open(base_metrics_path, 'r') as f:
                    all_data['base_model']['test_metrics'] = json.load(f)
    
    # Select experiment config based on type
    if experiment_type == 'lr':
        experiments = LEARNING_RATE_EXPERIMENTS
    elif experiment_type == 'optimizer':
        experiments = OPTIMIZER_EXPERIMENTS
    elif experiment_type == 'scheduler':
        experiments = SCHEDULER_EXPERIMENTS
    elif experiment_type == 'augmentation':
        experiments = AUGMENTATION_EXPERIMENTS
    else:
        return jsonify({'error': 'Invalid experiment type'})
    
    for exp_id, exp_config in experiments.items():
        try:
            exp_data = {
                'name': exp_config['name'],
                'training_time': None,
                'train_losses': [],
                'val_losses': [],
                'train_metrics': {},
                'val_metrics': {},
                'test_metrics': None
            }
            
            # Get training logs
            log_path = os.path.join(base_model_dir, exp_id, 'training_logs.json')
            print(f"Looking for logs at: {log_path}")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
                    exp_data['train_losses'] = logs.get('train_losses', [])
                    exp_data['val_losses'] = logs.get('val_losses', [])
                    exp_data['train_metrics'] = logs.get('train_metrics', {})
                    exp_data['val_metrics'] = logs.get('val_metrics', {})
                    total_time_minutes = sum(logs['epoch_times']) / 60
                    exp_data['training_time'] = round(total_time_minutes, 2)
                print(f"Loaded data for {exp_id}: {exp_data}")
            else:
                print(f"Log file not found at {log_path}")
            
            # Get test metrics
            metrics_path = os.path.join(base_model_dir, exp_id, 'test_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    exp_data['test_metrics'] = json.load(f)
            
            all_data[exp_id] = exp_data
            
        except Exception as e:
            print(f"Error processing experiment {exp_id}: {str(e)}")
            continue
    
    print(f"Returning data: {all_data}")
    return jsonify(all_data)

def get_model_metrics(model_id, is_benchmark=False):
    """Get metrics for a specific model from its test_metrics.json file"""
    # Determine the correct path based on whether it's a benchmark experiment
    if is_benchmark:
        metrics_path = os.path.join(LOGS_DIR, 'benchmark', model_id, 'test_metrics.json')
    else:
        metrics_path = os.path.join(LOGS_DIR, model_id, 'test_metrics.json')
    print(f"Looking for metrics at: {metrics_path}")

    if os.path.exists(metrics_path):
        print(f"Found metrics file for {model_id}")
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"Loaded metrics for {model_id}: {metrics}")
            return metrics  # Return the full metrics object
    else:
        print(f"No metrics file found for {model_id}")
    return {
        'IoU_Building': 0,
        'F1_Score_Building': 0,
        'Accuracy_Building': 0,
        'Precision_Building': 0,
        'Recall_Building': 0
    }

@app.route('/model-comparison')
def model_comparison():
    try:
        # Get base model config (UNet with best hyperparameters)
        base_config = EXPERIMENTS[BASE_MODEL_ID]
        
        # Get benchmark experiments
        benchmark_experiments = []
        for exp_id, exp_config in BENCHMARK_EXPERIMENTS.items():
            print(f"Adding experiment: {exp_id}, {exp_config['name']}")  # Debug print
            benchmark_experiments.append({
                'name': exp_config['name'],
                'id': exp_config['id']  # Make sure we're using the correct ID
            })
        
        print(f"Benchmark experiments: {benchmark_experiments}")  # Debug print
        
        # Get training time from logs
        log_path = os.path.join(LOGS_DIR, BASE_MODEL_ID, 'training_logs.json')
        training_time = 0
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
                training_time = sum(logs.get('epoch_times', [])) / 60
        
        metrics = get_model_metrics(BASE_MODEL_ID)
        metrics['training_time'] = round(training_time, 2)
        
        base_model = {
            'hyperparameters': {
                'learning_rate': TRAINING['LEARNING_RATE'],
                'optimizer': TRAINING['OPTIMIZER'],
                'scheduler': TRAINING['SCHEDULER'],
                'batch_size': TRAINING['BATCH_SIZE'],
                'epochs': TRAINING['NUM_EPOCHS']
            },
            'augmentations': TRAINING['AUGMENTATION']['TRAIN'],
            'metrics': metrics
        }
        
        return render_template('model_comparison.html', 
                             base_model=base_model,
                             experiments=benchmark_experiments,
                             base_metrics_json=json.dumps(metrics))
    except Exception as e:
        print(f"Error loading model comparison page: {str(e)}")
        empty_data = {
            'hyperparameters': {
                'learning_rate': 0,
                'optimizer': 'N/A',
                'scheduler': 'N/A',
                'batch_size': 0,
                'epochs': 0
            },
            'augmentations': {},
            'metrics': {
                'iou_building': 0,
                'f1_building': 0,
                'training_time': 0
            }
        }
        return render_template('model_comparison.html', 
                             base_model=empty_data,
                             base_metrics_json=json.dumps(empty_data['metrics']))

@app.route('/get_model_comparison_data')
def get_model_comparison_data():
    experiment_type = request.args.get('type', 'default')
    print(f"\nFetching data for experiment type: {experiment_type}")
    
    if experiment_type == 'benchmark':
        experiments_config = BENCHMARK_EXPERIMENTS
        base_path = os.path.join(LOGS_DIR, 'benchmark')
        print(f"Using benchmark experiments: {list(experiments_config.keys())}")
        print(f"Looking in path: {base_path}")
    else:
        experiments_config = EXPERIMENTS
        base_path = LOGS_DIR
    
    data = {}
    for exp_id, exp_config in experiments_config.items():
        exp_log_path = os.path.join(base_path, exp_id, 'training_logs.json')
        print(f"\nChecking for logs at: {exp_log_path}")
        if os.path.exists(exp_log_path):
            print(f"Found logs for {exp_id}")
            with open(exp_log_path, 'r') as f:
                log_data = json.load(f)
                data[exp_id] = {
                    'name': exp_config['name'],
                    'train_losses': log_data['train_losses'],
                    'val_losses': log_data['val_losses'],
                    'train_metrics': log_data['train_metrics'],
                    'val_metrics': log_data['val_metrics'],
                    'test_metrics': get_model_metrics(exp_id, experiment_type == 'benchmark')
                }
        else:
            print(f"No logs found for {exp_id}")
    
    print(f"\nReturning data for models: {list(data.keys())}")
    return jsonify(data)

if __name__ == '__main__':
    app.run(
        host=WEBAPP['HOST'],
        port=WEBAPP['PORT'],
        debug=WEBAPP['DEBUG']
    ) 