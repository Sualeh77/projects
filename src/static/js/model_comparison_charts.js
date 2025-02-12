document.addEventListener('DOMContentLoaded', function() {
    const updateButton = document.getElementById('updateModelCharts');
    
    if (updateButton) {
        updateButton.addEventListener('click', updateModelCharts);
    }
    
    // Initial load
    updateModelCharts();
});

function updateModelCharts() {
    fetch('/get_model_comparison_data')
        .then(response => response.json())
        .then(data => {
            createExperimentCharts(data, 'model', 'Model Architecture');
            updateMetricsTable(data, window.baseModelMetrics, 'modelMetricsTable');
        })
        .catch(error => console.error('Error:', error));
} 