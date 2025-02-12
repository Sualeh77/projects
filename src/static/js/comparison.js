function createComparisonCharts(experiments) {
    if (!experiments || Object.keys(experiments).length === 0) {
        console.error('No experiments data available');
        return;
    }

    const experimentIds = Object.keys(experiments);
    
    // Building Metrics Comparison Chart
    const metrics = ['IoU', 'F1_Score', 'Accuracy', 'Precision', 'Recall'];
    const buildingData = experimentIds.map(id => ({
        name: experiments[id]?.name || id,
        values: {
            'IoU': experiments[id]?.test_metrics?.['IoU_Building'] || 0,
            'F1_Score': experiments[id]?.test_metrics?.['F1_Score_Building'] || 0,
            'Accuracy': experiments[id]?.test_metrics?.['Accuracy_Building'] || 0,
            'Precision': experiments[id]?.test_metrics?.['Precision_Building'] || 0,
            'Recall': experiments[id]?.test_metrics?.['Recall_Building'] || 0
        }
    }));

    // Create traces for each model
    const traces = buildingData.map(model => ({
        x: metrics,
        y: metrics.map(metric => model.values[metric]),
        name: model.name,
        type: 'bar'
    }));

    // Calculate min and max values for auto-ranging
    const allValues = buildingData.flatMap(model => 
        Object.values(model.values)
    ).filter(val => val > 0);
    const minValue = Math.floor(Math.min(...allValues) * 1000) / 1000; // Round down to 3 decimals
    const maxValue = Math.ceil(Math.max(...allValues) * 1000) / 1000;  // Round up to 3 decimals
    
    // Add small padding to the ranges (0.2% of the range)
    const range = maxValue - minValue;
    const padding = range * 0.002;
    const paddedMin = Math.max(0, minValue - padding);
    const paddedMax = Math.min(1, maxValue + padding);

    // Create the chart
    const chart = Plotly.newPlot('buildingMetricsChart', traces, {
        title: 'Building Segmentation Performance Metrics',
        barmode: 'group',
        xaxis: { 
            title: 'Metrics',
            tickangle: 0,
            titlefont: {
                size: 14
            },
            tickfont: {
                size: 12
            }
        },
        yaxis: { 
            title: 'Score',
            range: [0.8, 1.0],
            tickformat: '.4f'
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: -0.4,  // Move legend further down
            xanchor: 'center',
            x: 0.5,   // Center the legend
            font: {
                size: 12
            }
        },
        margin: {
            l: 80,   // left margin
            r: 50,   // right margin
            t: 50,   // top margin
            b: 150   // Increase bottom margin to accommodate legend and axis labels
        },
        annotations: [{
            text: 'Double-click to toggle zoom',
            xref: 'paper',
            yref: 'paper',
            x: 1,
            xanchor: 'right',
            y: 1,
            yanchor: 'bottom',
            showarrow: false,
            font: {
                size: 12,
                color: 'gray'
            }
        }]
    });

    // Add double-click handler for zoom toggle
    let isZoomed = true;
    document.getElementById('buildingMetricsChart').on('dblclick', function() {
        isZoomed = !isZoomed;
        Plotly.relayout('buildingMetricsChart', {
            'yaxis.range': isZoomed ? 
                [0.8, 1.0] : // Zoomed range
                [paddedMin, paddedMax] // Full range using actual min/max values with padding
        });
    });

    // Model Complexity Chart
    const complexityData = {
        labels: experimentIds.map(id => experiments[id]?.name || id),
        values: experimentIds.map(id => ({
            total: experiments[id]?.model_parameters?.total || 0,
            encoder: experiments[id]?.model_parameters?.encoder || 0,
            decoder: experiments[id]?.model_parameters?.decoder || 0
        }))
    };

    // Helper function to get encoder name from full name
    function getEncoderName(fullName) {
        // Extract encoder name (e.g., "ResNet18" from "UNet with ResNet18 Encoder")
        const match = fullName.match(/with\s+(\w+)\s+Encoder/);
        return match ? match[1] : fullName;
    }

    // Create traces for complexity and efficiency charts
    const complexityTrace = {
        x: complexityData.labels.map(name => getEncoderName(name)),
        y: complexityData.values.map(v => v.total),
        type: 'bar',
        name: 'Total Parameters',
        marker: {
            color: '#2196F3',
            line: { color: '#1976D2', width: 1 }
        }
    };

    const efficiencyData = {
        labels: experimentIds.map(id => experiments[id].name),
        values: experimentIds.map(id => experiments[id].training_time || 0)
    };

    const efficiencyTrace = {
        x: efficiencyData.labels.map(name => getEncoderName(name)),
        y: efficiencyData.values,
        type: 'bar',
        name: 'Training Time',
        marker: {
            color: '#4CAF50',
            line: { color: '#388E3C', width: 1 }
        }
    };

    // Create a div for side-by-side sections
    const complexityTimeContainer = document.createElement('div');
    complexityTimeContainer.style.display = 'flex';
    complexityTimeContainer.style.gap = '20px';
    complexityTimeContainer.style.marginBottom = '25px';

    // Create sections for each chart
    const complexitySection = document.createElement('div');
    complexitySection.className = 'comparison-card';
    complexitySection.style.flex = '1';
    complexitySection.innerHTML = `
        <h3>Model Complexity</h3>
        <div id="modelComplexityChart"></div>
    `;

    const timeSection = document.createElement('div');
    timeSection.className = 'comparison-card';
    timeSection.style.flex = '1';
    timeSection.innerHTML = `
        <h3>Training Efficiency</h3>
        <div id="trainingEfficiencyChart"></div>
    `;

    // Add sections to container
    complexityTimeContainer.appendChild(complexitySection);
    complexityTimeContainer.appendChild(timeSection);

    // Insert container after the building metrics chart section
    document.querySelector('.comparison-card').insertAdjacentElement('afterend', complexityTimeContainer);

    // Plot complexity chart
    Plotly.newPlot('modelComplexityChart', [complexityTrace], {
        title: 'UNet Model Parameters by Encoder',
        yaxis: { 
            title: 'Number of Parameters',
            tickformat: '.2s'
        },
        margin: { t: 40, l: 80, r: 20, b: 60 },
        height: 350,
        xaxis: {
            tickangle: -45
        }
    });

    // Plot efficiency chart
    Plotly.newPlot('trainingEfficiencyChart', [efficiencyTrace], {
        title: 'UNet Training Time by Encoder',
        yaxis: { 
            title: 'Minutes',
            tickformat: '.0f'
        },
        margin: { t: 40, l: 60, r: 20, b: 60 },
        height: 350,
        xaxis: {
            tickangle: -45
        }
    });

    // Convergence Analysis Chart
    const convergenceData = experimentIds.map(id => ({
        x: Array.from({length: experiments[id].train_losses.length}, (_, i) => i + 1),
        y: experiments[id].train_losses,
        name: experiments[id].name,
        type: 'scatter',
        mode: 'lines'
    }));

    Plotly.newPlot('convergenceChart', convergenceData, {
        title: 'Training Loss Convergence',
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' }
    });

    // Generate Conclusion only if we have valid data
    if (experimentIds.some(id => experiments[id]?.test_metrics?.IoU_Building)) {
        generateConclusion(experiments);
    }
}

function generateConclusion(experiments) {
    const experimentIds = Object.keys(experiments).filter(id => 
        experiments[id]?.test_metrics?.IoU_Building !== undefined
    );
    
    if (experimentIds.length === 0) {
        document.getElementById('modelConclusion').innerHTML = 
            '<p>No test metrics available for comparison yet.</p>';
        return;
    }

    // Find best performing model based on building IoU
    const bestModel = experimentIds.reduce((best, id) => {
        const iou = experiments[id]?.test_metrics?.IoU_Building || 0;
        return iou > best.metrics ? 
            {id, metrics: iou} : best;
    }, {id: null, metrics: 0});

    // Find most efficient model (best performance/parameter ratio)
    const mostEfficient = experimentIds.reduce((best, id) => {
        const iou = experiments[id]?.test_metrics?.IoU_Building || 0;
        const params = experiments[id]?.model_parameters?.total || 1;
        const efficiency = iou / params;
        return efficiency > best.efficiency ? 
            {id, efficiency} : best;
    }, {id: null, efficiency: 0});

    // Only generate conclusion if we have valid models
    if (!bestModel.id || !mostEfficient.id) {
        document.getElementById('modelConclusion').innerHTML = 
            '<p>Insufficient data to generate conclusions.</p>';
        return;
    }

    const conclusion = `
        <p><strong>Performance Analysis:</strong></p>
        <ul>
            <li>Best Overall Performance: ${experiments[bestModel.id]?.name || 'Unknown'} achieved the highest building IoU of ${bestModel.metrics.toFixed(4)}</li>
            <li>Most Efficient Model: ${experiments[mostEfficient.id]?.name || 'Unknown'} provides the best performance per parameter ratio</li>
        </ul>
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Deeper models (ResNet50, ResNet101) showed ${experiments[bestModel.id]?.name?.includes('ResNet50') ? 'better' : 'worse'} performance but required more computational resources</li>
            <li>Training time increased significantly with model complexity</li>
            <li>The sweet spot between performance and efficiency appears to be ${experiments[mostEfficient.id]?.name || 'Unknown'}</li>
        </ul>
        <p><strong>Recommendation:</strong> Based on the comprehensive analysis, ${experiments[bestModel.id]?.name || 'Unknown'} Encoder is recommended as the benchmark model for comparing various approaches to building footprint extraction from satellite imagery, offering the best balance of accuracy and practical usability. While its complexity is significantly higher than that of UNet with ResNet18, and the performance metrics show only slight differences, comparing a UNet-based model with Transformer-based models, which have more parameters, would make the comparison between UNet with ResNet18 and larger Transformer models unfair. Therefore, ${experiments[bestModel.id]?.name || 'Unknown'} is chosen as the benchmark model for further hyperparameter tuning.</p>
    `;

    document.getElementById('modelConclusion').innerHTML = conclusion;
} 