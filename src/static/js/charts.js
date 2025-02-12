// Store charts for each experiment
const charts = {};

function createCharts(exp_id) {
    console.log(`Creating charts for experiment: ${exp_id}`);
    
    // Training Metrics Chart
    const trainingCtx = document.getElementById(`trainingChart-${exp_id}`);
    if (!trainingCtx) {
        console.error(`Training chart element not found for ${exp_id}`);
        return;
    }

    charts[`training-${exp_id}`] = Plotly.newPlot(`trainingChart-${exp_id}`, [{
        y: [],
        mode: 'lines',
        name: 'Training Loss',
        line: { color: 'rgb(219, 64, 82)' }
    }, {
        y: [],
        mode: 'lines',
        name: 'Training IoU',
        yaxis: 'y2',
        line: { color: 'rgb(64, 145, 219)' }
    }], {
        title: 'Training Metrics',
        xaxis: { title: 'Epoch' },
        yaxis: { 
            title: 'Loss',
            titlefont: { color: 'rgb(219, 64, 82)' },
            tickfont: { color: 'rgb(219, 64, 82)' }
        },
        yaxis2: {
            title: 'IoU',
            titlefont: { color: 'rgb(64, 145, 219)' },
            tickfont: { color: 'rgb(64, 145, 219)' },
            overlaying: 'y',
            side: 'right'
        },
        showlegend: true
    });

    // Validation Metrics Chart
    const validationCtx = document.getElementById(`validationChart-${exp_id}`);
    if (!validationCtx) {
        console.error(`Validation chart element not found for ${exp_id}`);
        return;
    }

    charts[`validation-${exp_id}`] = Plotly.newPlot(`validationChart-${exp_id}`, [{
        y: [],
        mode: 'lines',
        name: 'Validation Loss',
        line: { color: 'rgb(219, 64, 82)' }
    }, {
        y: [],
        mode: 'lines',
        name: 'Validation IoU',
        yaxis: 'y2',
        line: { color: 'rgb(64, 145, 219)' }
    }], {
        title: 'Validation Metrics',
        xaxis: { title: 'Epoch' },
        yaxis: { 
            title: 'Loss',
            titlefont: { color: 'rgb(219, 64, 82)' },
            tickfont: { color: 'rgb(219, 64, 82)' }
        },
        yaxis2: {
            title: 'IoU',
            titlefont: { color: 'rgb(64, 145, 219)' },
            tickfont: { color: 'rgb(64, 145, 219)' },
            overlaying: 'y',
            side: 'right'
        },
        showlegend: true
    });
}

function updateTrainingMetrics(exp_id) {
    console.log(`Updating training metrics for: ${exp_id}`);
    
    fetch(`/get_training_logs/${exp_id}`)
        .then(response => response.json())
        .then(data => {
            console.log(`Received training data for ${exp_id}:`, data);
            
            if (data.error) {
                console.error('Error:', data.error);
                return;
            }
            
            // Update summary metrics
            const elements = {
                'best-train-loss': data.best_train_loss,
                'best-val-loss': data.best_val_loss,
                'avg-epoch-time': data.avg_epoch_time ? `${data.avg_epoch_time}s` : null,
                'total-train-time': data.total_train_time ? `${data.total_train_time}m` : null,
                'total-epochs': data.current_epoch
            };

            for (const [key, value] of Object.entries(elements)) {
                const element = document.getElementById(`${key}-${exp_id}`);
                if (element) {
                    element.textContent = value ? value.toString() : '-';
                } else {
                    console.error(`Element not found: ${key}-${exp_id}`);
                }
            }

            // Update charts
            if (data.train_losses && data.val_losses) {
                const epochs = Array.from({length: data.train_losses.length}, (_, i) => i + 1);
                
                try {
                    // Update Training Metrics Chart
                    if (data.train_metrics && data.train_metrics.IoU_all) {
                        console.log("Updating training chart with:", {
                            losses: data.train_losses,
                            iou: data.train_metrics.IoU_all
                        });
                        
                        Plotly.update(`trainingChart-${exp_id}`, {
                            x: [epochs, epochs],
                            y: [data.train_losses, data.train_metrics.IoU_all]
                        });
                    }
                    
                    // Update Validation Metrics Chart
                    if (data.val_metrics && data.val_metrics.IoU_all) {
                        console.log("Updating validation chart with:", {
                            losses: data.val_losses,
                            iou: data.val_metrics.IoU_all
                        });
                        
                        Plotly.update(`validationChart-${exp_id}`, {
                            x: [epochs, epochs],
                            y: [data.val_losses, data.val_metrics.IoU_all]
                        });
                    }
                } catch (error) {
                    console.error(`Error updating charts for ${exp_id}:`, error);
                }
            }
        })
        .catch(error => console.error(`Error fetching training data for ${exp_id}:`, error));
}

function updateTestMetrics(exp_id) {
    console.log(`Updating test metrics for: ${exp_id}`);
    
    fetch(`/get_test_metrics/${exp_id}`)
        .then(response => response.json())
        .then(data => {
            console.log(`Received test metrics for ${exp_id}:`, data);
            
            if (data.error) {
                console.error('Error:', data.error);
                return;
            }
            
            // Update all metrics
            const metrics = ['iou', 'f1-score', 'accuracy', 'precision', 'recall'];
            const classes = ['all', 'building', 'boundary', 'contact'];
            
            metrics.forEach(metric => {
                classes.forEach(cls => {
                    const elementId = `${metric}-${cls}-${exp_id}`;
                    const element = document.getElementById(elementId);
                    if (element) {
                        const value = data[metric] && data[metric][cls] 
                            ? data[metric][cls].toFixed(4) 
                            : '-';
                        element.textContent = value;
                    } else {
                        console.error(`Element not found: ${elementId}`);
                    }
                });
            });
        })
        .catch(error => console.error(`Error fetching test metrics for ${exp_id}:`, error));
}

// Initialize charts for each experiment when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing experiments...');
    
    // Find all experiment cards and initialize their charts
    const experimentCards = document.querySelectorAll('.experiment-card');
    console.log(`Found ${experimentCards.length} experiment cards`);
    
    experimentCards.forEach(card => {
        const metricElement = card.querySelector('[id^="best-train-loss-"]');
        if (metricElement) {
            const exp_id = metricElement.id.split('-').pop();
            console.log(`Initializing experiment: ${exp_id}`);
            createCharts(exp_id);
            updateTrainingMetrics(exp_id);
            updateTestMetrics(exp_id);
        } else {
            console.error('Could not find metric element in experiment card');
        }
    });

    // Initialize comparison section
    updateComparisonSection();
});

function createChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Value'
                    }
                }
            }
        }
    });
}

function updateCharts(experimentName) {
    console.log("Updating charts for experiment:", experimentName);
    
    // Fetch training logs
    fetch(`/get_training_logs/${experimentName}`)
        .then(response => response.json())
        .then(data => {
            console.log("Received training data:", data);
            if (Object.keys(data).length > 0) {
                updateTrainingCharts(data);
            }
        })
        .catch(error => console.error('Error fetching training logs:', error));
    
    // Fetch test metrics
    fetch(`/get_test_metrics/${experimentName}`)
        .then(response => response.json())
        .then(data => {
            console.log("Received test metrics:", data);
            if (Object.keys(data).length > 0) {
                updateTestMetrics(data);
            }
        })
        .catch(error => console.error('Error fetching test metrics:', error));
}

function updateTrainingCharts(data) {
    // Update loss chart
    const epochs = Array.from({length: data.train_losses.length}, (_, i) => i + 1);
    
    // Loss chart
    const lossChart = document.getElementById('lossChart');
    if (lossChart) {
        Plotly.newPlot('lossChart', [
            {
                x: epochs,
                y: data.train_losses,
                name: 'Training Loss',
                type: 'scatter'
            },
            {
                x: epochs,
                y: data.val_losses,
                name: 'Validation Loss',
                type: 'scatter'
            }
        ], {
            title: 'Training and Validation Loss',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' }
        });
    }
    
    // Metrics charts
    if (data.train_metrics) {
        Object.keys(data.train_metrics).forEach(metric => {
            const chartId = `${metric}Chart`;
            const chartElement = document.getElementById(chartId);
            if (chartElement) {
                Plotly.newPlot(chartId, [
                    {
                        x: epochs,
                        y: data.train_metrics[metric],
                        name: `Training ${metric}`,
                        type: 'scatter'
                    },
                    {
                        x: epochs,
                        y: data.val_metrics[metric],
                        name: `Validation ${metric}`,
                        type: 'scatter'
                    }
                ], {
                    title: `Training and Validation ${metric}`,
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: metric }
                });
            }
        });
    }
}

// Add manual update buttons
function addUpdateButtons() {
    const controlPanel = document.createElement('div');
    controlPanel.className = 'control-panel';
    controlPanel.innerHTML = `
        <button onclick="updateTrainingMetrics('unet_resnet18')">Update Training Metrics</button>
        <button onclick="updateTestMetrics('unet_resnet18')">Update Test Metrics</button>
    `;
    document.querySelector('main').prepend(controlPanel);
}

function updateComparisonSection() {
    fetch('/get_all_experiments_data')
        .then(response => response.json())
        .then(data => {
            console.log('Received experiments data:', data);
            createComparisonCharts(data);
        })
        .catch(error => console.error('Error fetching experiments data:', error));
} 