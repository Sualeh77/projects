function createExperimentCharts(data, containerId, title) {
    console.log(`Creating charts for ${title}:`, data);
    const experiments = Object.values(data);
    console.log('Experiments:', experiments);
    
    // Create training metrics traces
    const trainingTraces = experiments.map(exp => {
        console.log('Processing experiment:', exp);
        const isBaseModel = exp.name === 'Base Model';
        if (!exp.train_losses) {
            return [];
        }
        return [
            {
                x: Array.from({ length: exp.train_losses.length }, (_, i) => i + 1),
                y: exp.train_losses,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Loss)`,
                hovertemplate: 'Epoch: %{x}<br>Loss: %{y:.4f}',
                yaxis: 'y1',
                line: {
                    dash: isBaseModel ? 'dash' : 'solid',
                    width: isBaseModel ? 3 : 2
                }
            }
        ];
    }).flat();

    // Add training IoU to training traces
    experiments.forEach(exp => {
        const isBaseModel = exp.name === 'Base Model';
        const trainIoU = exp.train_metrics?.['IoU_all'] || exp.train_metrics?.IoU_Building || [];
        if (trainIoU.length > 0) {
            trainingTraces.push({
                x: Array.from({ length: trainIoU.length }, (_, i) => i + 1),
                y: trainIoU,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (IoU)`,
                hovertemplate: 'Epoch: %{x}<br>IoU: %{y:.4f}',
                yaxis: 'y2',
                line: {
                    dash: isBaseModel ? 'dash' : 'solid',
                    width: isBaseModel ? 3 : 2
                }
            });
        }
    });

    // Create validation metrics traces
    const validationTraces = experiments.map(exp => {
        const isBaseModel = exp.name === 'Base Model';
        if (!exp.val_losses) {
            return [];
        }
        return [
            {
                x: Array.from({ length: exp.val_losses.length }, (_, i) => i + 1),
                y: exp.val_losses,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Loss)`,
                hovertemplate: 'Epoch: %{x}<br>Loss: %{y:.4f}',
                yaxis: 'y1',
                line: {
                    dash: isBaseModel ? 'dash' : 'solid',
                    width: isBaseModel ? 3 : 2
                }
            }
        ];
    }).flat();

    // Add validation IoU to validation traces
    experiments.forEach(exp => {
        const isBaseModel = exp.name === 'Base Model';
        const valIoU = exp.val_metrics?.['IoU_all'] || exp.val_metrics?.IoU_Building || [];
        if (valIoU.length > 0) {
            validationTraces.push({
                x: Array.from({ length: valIoU.length }, (_, i) => i + 1),
                y: valIoU,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (IoU)`,
                hovertemplate: 'Epoch: %{x}<br>IoU: %{y:.4f}',
                yaxis: 'y2',
                line: {
                    dash: isBaseModel ? 'dash' : 'solid',
                    width: isBaseModel ? 3 : 2
                }
            });
        }
    });

    // Create training metrics layout
    const trainingLayout = {
        title: `${title} - Training Metrics`,
        xaxis: { title: 'Epoch' },
        yaxis: { 
            title: 'Loss',
            side: 'left'
        },
        yaxis2: {
            title: 'IoU',
            side: 'right',
            overlaying: 'y'
        },
        showlegend: true,
        legend: { x: 1.1, xanchor: 'left', y: 1 }
    };

    // Create validation metrics layout
    const validationLayout = {
        title: `${title} - Validation Metrics`,
        xaxis: { title: 'Epoch' },
        yaxis: { 
            title: 'Loss',
            side: 'left'
        },
        yaxis2: {
            title: 'IoU',
            side: 'right',
            overlaying: 'y'
        },
        showlegend: true,
        legend: { x: 1.1, xanchor: 'left', y: 1 }
    };

    // Plot the charts
    if (trainingTraces.length > 0) {
        Plotly.newPlot(`${containerId}TrainingChart`, trainingTraces, trainingLayout);
    }
    if (validationTraces.length > 0) {
        Plotly.newPlot(`${containerId}ValidationChart`, validationTraces, validationLayout);
    }
}

function updateMetricsTable(data, baseModelMetrics, tableId) {
    const table = document.getElementById(tableId);
    const thead = table.querySelector('thead tr');
    const tbody = table.querySelectorAll('tbody tr');
    
    // Clear existing headers except first column
    while (thead.children.length > 1) {
        thead.removeChild(thead.lastChild);
    }
    
    // Add base model column
    thead.insertCell().textContent = 'Base Model';
    
    // Add experiment columns
    Object.entries(data).forEach(([expId, exp]) => {
        // Skip base model as it's already added
        if (exp.name === 'Base Model') return;
        thead.insertCell().textContent = exp.name;
    });
    
    // Add "Best Model" column
    thead.insertCell().textContent = 'Best Model';
    
    // Helper function to find best model for a metric
    const findBestModel = (metricValues) => {
        let bestValue = -Infinity;
        let bestModel = '';
        Object.entries(metricValues).forEach(([model, value]) => {
            if (value > bestValue) {
                bestValue = value;
                bestModel = model;
            }
        });
        return bestModel;
    };
    
    // Update metrics rows
    const metrics = ['IoU', 'F1-Score', 'Accuracy', 'Precision', 'Recall'];
    const metricKeys = {
        'IoU': 'IoU_Building',
        'F1-Score': 'F1_Score_Building',
        'Accuracy': 'Accuracy_Building',
        'Precision': 'Precision_Building',
        'Recall': 'Recall_Building'
    };
    
    metrics.forEach((metric, idx) => {
        const row = tbody[idx];
        // Clear existing cells except first column
        while (row.children.length > 1) {
            row.removeChild(row.lastChild);
        }
        
        // Add base model value
        const baseValue = baseModelMetrics[metricKeys[metric]] || 0;
        row.insertCell().textContent = baseValue.toFixed(4);
        
        // Create map for finding best model
        const metricValues = { 'Base Model': baseValue };
        
        // Add experiment values
        Object.entries(data).forEach(([expId, exp]) => {
            // Skip base model as it's already added
            if (exp.name === 'Base Model') return;
            const value = exp.test_metrics?.[metricKeys[metric]] || 0;
            row.insertCell().textContent = value.toFixed(4);
            metricValues[exp.name] = value;
        });
        
        // Add best model name
        const bestModel = findBestModel(metricValues);
        row.insertCell().textContent = bestModel;
    });
}

function updateLearningRateCharts() {
    fetch('/get_hyperparameter_experiments?type=lr')
        .then(response => response.json())
        .then(data => {
            console.log('Received LR data:', data);
            createExperimentCharts(data, 'lr', 'Learning Rate');
            updateMetricsTable(data, window.baseModelMetrics, 'lrMetricsTable');
        })
        .catch(error => console.error('Error:', error));
}

function updateOptimizerCharts() {
    fetch('/get_hyperparameter_experiments?type=optimizer')
        .then(response => response.json())
        .then(data => {
            createExperimentCharts(data, 'optimizer', 'Optimizer');
            updateMetricsTable(data, window.baseModelMetrics, 'optimizerMetricsTable');
        })
        .catch(error => console.error('Error:', error));
}

function updateSchedulerCharts() {
    fetch('/get_hyperparameter_experiments?type=scheduler')
        .then(response => response.json())
        .then(data => {
            createExperimentCharts(data, 'scheduler', 'Scheduler');
            updateMetricsTable(data, window.baseModelMetrics, 'schedulerMetricsTable');
        })
        .catch(error => console.error('Error:', error));
}

function updateAugmentationCharts() {
    fetch('/get_hyperparameter_experiments?type=augmentation')
        .then(response => response.json())
        .then(data => {
            createExperimentCharts(data, 'augmentation', 'Augmentation');
            updateMetricsTable(data, window.baseModelMetrics, 'augmentationMetricsTable');
        })
        .catch(error => console.error('Error:', error));
}

// Move all initialization code inside DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for update buttons
    const lrButton = document.getElementById('updateLRCharts');
    const optimizerButton = document.getElementById('updateOptimizerCharts');
    const schedulerButton = document.getElementById('updateSchedulerCharts');
    const augmentationButton = document.getElementById('updateAugmentationCharts');
    
    if (lrButton) {
        lrButton.addEventListener('click', updateLearningRateCharts);
    }
    if (optimizerButton) {
        optimizerButton.addEventListener('click', updateOptimizerCharts);
    }
    if (schedulerButton) {
        schedulerButton.addEventListener('click', updateSchedulerCharts);
    }
    if (augmentationButton) {
        augmentationButton.addEventListener('click', updateAugmentationCharts);
    }
    
    // Initial load
    updateLearningRateCharts();
    updateOptimizerCharts();
    updateSchedulerCharts();
    updateAugmentationCharts();
}); 