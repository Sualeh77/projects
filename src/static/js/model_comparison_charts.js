let currentTraces = [];
let currentVisibility = 'both';

function createExperimentCharts(data, containerId, title) {
    console.log(`Creating charts for ${title}:`, data);
    const experiments = Object.values(data);
    
    // Create combined traces for all models
    currentTraces = [];
    
    // Different colors for different models
    const colors = ['#1f77b4', '#ff7f0e', '#8c564b', '#2ca02c', '#9467bd', '#d62728'];  // Blue, Orange, Brown
    
    experiments.forEach((exp, index) => {
        console.log(`Creating traces for model: ${exp.name}, ID: ${exp.id}`);
        // Add loss traces
        currentTraces.push(
            {
                x: Array.from({ length: exp.train_losses.length }, (_, i) => i + 1),
                y: exp.train_losses,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Train Loss)`,
                hovertemplate: 'Epoch: %{x}<br>Loss: %{y:.4f}',
                yaxis: 'y1',
                isTraining: true,
                modelId: exp.id,
                line: {
                    dash: 'solid',
                    width: 2,
                    color: colors[index]
                },
                showlegend: false
            },
            {
                x: Array.from({ length: exp.val_losses.length }, (_, i) => i + 1),
                y: exp.val_losses,
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Val Loss)`,
                hovertemplate: 'Epoch: %{x}<br>Loss: %{y:.4f}',
                yaxis: 'y1',
                isTraining: false,
                modelId: exp.id,
                line: {
                    dash: 'dot',
                    width: 2,
                    color: colors[index]
                },
                showlegend: false
            }
        );
        
        // Add IoU traces
        currentTraces.push(
            {
                x: Array.from({ length: exp.train_metrics.IoU_Building?.length || 0 }, (_, i) => i + 1),
                y: exp.train_metrics.IoU_Building || [],
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Train IoU)`,
                hovertemplate: 'Epoch: %{x}<br>IoU: %{y:.4f}',
                yaxis: 'y2',
                isTraining: true,
                modelId: exp.id,
                line: { 
                    color: colors[index],
                    dash: 'solid',
                    width: 2
                }
            },
            {
                x: Array.from({ length: exp.val_metrics.IoU_Building?.length || 0 }, (_, i) => i + 1),
                y: exp.val_metrics.IoU_Building || [],
                type: 'scatter',
                mode: 'lines',
                name: `${exp.name} (Val IoU)`,
                hovertemplate: 'Epoch: %{x}<br>IoU: %{y:.4f}',
                yaxis: 'y2',
                isTraining: false,
                modelId: exp.id,
                line: { 
                    color: colors[index],
                    dash: 'dot',
                    width: 2
                }
            }
        );
    });
    
    // First plot the chart
    plotChart();
    // Then update visibility based on current selection
    updateChartVisibility();
}

function updateChartVisibility() {
    const visibility = currentTraces.map(trace => {
        // Check if training/validation state is visible
        const isStateVisible = currentVisibility === 'both' || 
            (currentVisibility === 'train' && trace.isTraining) ||
            (currentVisibility === 'val' && !trace.isTraining);
        
        return isStateVisible;
    });
    
    // Update traces visibility
    Plotly.restyle('modelComparisonChart', {
        'visible': visibility
    });
}

function updateMetricsTable(data, baseMetrics, tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    console.log('Updating metrics table with data:', data);

    // Get all experiment names
    const experiments = Object.keys(data);

    // Update headers
    const headerRow = table.querySelector('thead tr');
    headerRow.innerHTML = '<th>Metric</th>';
    experiments.forEach(exp => {
        headerRow.innerHTML += `<th>${data[exp].name}</th>`;
    });

    // Update metrics
    const metrics = [
        {display: 'IoU', key: 'IoU_Building'},
        {display: 'F1-Score', key: 'F1_Score_Building'},
        {display: 'Accuracy', key: 'Accuracy_Building'},
        {display: 'Precision', key: 'Precision_Building'},
        {display: 'Recall', key: 'Recall_Building'}
    ];
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';

    metrics.forEach(metric => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${metric.display}</td>`;
        
        experiments.forEach(exp => {
            const value = data[exp].test_metrics?.[metric.key] || 0;
            row.innerHTML += `<td>${value.toFixed(4)}</td>`;
        });
        
        tbody.appendChild(row);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const updateButton = document.getElementById('updateModelCharts');
    const toggleButtons = document.querySelectorAll('.toggle-btn[data-show]');
    
    if (updateButton) {
        updateButton.addEventListener('click', updateModelCharts);
    }
    
    toggleButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            toggleButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            currentVisibility = this.dataset.show;
            updateChartVisibility();
        });
    });
    
    // Initial load
    updateModelCharts();
});

function updateModelCharts() {
    fetch('/get_model_comparison_data?type=benchmark')
        .then(response => response.json())
        .then(data => {
            createExperimentCharts(data, 'model', 'Model Architecture');
            updateMetricsTable(data, window.baseModelMetrics, 'modelMetricsTable');
        })
        .catch(error => console.error('Error:', error));
}

function plotChart() {
    // Initial plot of the chart
    Plotly.newPlot('modelComparisonChart', currentTraces, {
        title: 'Model Architecture Comparison',
        xaxis: { 
            title: 'Epoch',
            titlefont: { size: 14 },
            titlepadding: 20
        },
        yaxis: { 
            title: 'Loss',
            titlefont: {color: '#1f77b4'},
            tickfont: {color: '#1f77b4'}
        },
        yaxis2: {
            title: 'IoU',
            titlefont: {color: '#2ca02c'},
            tickfont: {color: '#2ca02c'},
            overlaying: 'y',
            side: 'right',
            range: [0, 1]
        },
        margin: {
            l: 60,
            r: 60,
            b: 60,
            t: 40
        }
    });
} 