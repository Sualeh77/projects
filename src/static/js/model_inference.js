class ModelInference {
    constructor() {
        this.modelColors = {
            'unet_resnet101_benchmark': 'rgba(255, 0, 0, 0.5)',    // Red
            'unetpp_resnet50_benchmark': 'rgba(0, 255, 0, 0.5)',   // Green
            'deeplabv3plus_resnet101_benchmark': 'rgba(0, 0, 255, 0.5)', // Blue
            'segformer_b3_benchmark': 'rgba(255, 165, 0, 0.5)',    // Orange
            'clipseg': 'rgba(255, 0, 255, 0.5)',                    // Magenta
            'sam2': 'rgba(0, 0, 0, 0.5)'                            // Black
        };
        this.predictions = new Map();
        this.setupEventListeners();
    }

    setupEventListeners() {
        const runButton = document.getElementById('runInference');
        runButton.addEventListener('click', () => this.runInference());
        
        // Enable run button when both files are uploaded
        document.getElementById('imageUpload').addEventListener('change', () => this.checkRunButton());
        document.getElementById('maskUpload').addEventListener('change', () => this.checkRunButton());
    }

    checkRunButton() {
        const imageFile = document.getElementById('imageUpload').files[0];
        const maskFile = document.getElementById('maskUpload').files[0];
        document.getElementById('runInference').disabled = !(imageFile && maskFile);
    }

    async runInference() {
        const imageFile = document.getElementById('imageUpload').files[0];
        const maskFile = document.getElementById('maskUpload').files[0];
        
        if (!imageFile || !maskFile) return;

        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('mask', maskFile);

        try {
            const response = await fetch('/run_inference', {
                method: 'POST',
                body: formData
            });
            
            const results = await response.json();
            if (results.error) throw new Error(results.error);
            
            this.displayResults(results);
        } catch (error) {
            console.error('Inference failed:', error);
            alert('Failed to run inference: ' + error.message);
        }
    }

    displayResults(results) {
        const overlaysContainer = document.getElementById('predictionOverlays');
        const togglesContainer = document.querySelector('.toggle-group');
        const metricsContainer = document.getElementById('iouMetrics');
        
        // Clear previous results
        overlaysContainer.innerHTML = '';
        // Clear only model toggles, keep ground truth toggle
        const modelToggles = togglesContainer.querySelectorAll('.toggle-item:not(:first-child)');
        modelToggles.forEach(toggle => toggle.remove());
        metricsContainer.innerHTML = '';
        
        // Create metrics table
        const table = document.createElement('table');
        
        // Add results for each model
        Object.entries(results.predictions).forEach(([modelName, data]) => {
            // Add mask overlay
            const overlay = document.createElement('img');
            overlay.src = data.mask_url;
            overlay.className = 'model-mask';
            overlay.style.filter = `opacity(0.7) drop-shadow(0 0 0 ${this.modelColors[modelName]})`;
            overlay.style.display = 'none';
            overlaysContainer.appendChild(overlay);
            
            // Add toggle checkbox
            const toggleDiv = document.createElement('div');
            toggleDiv.className = 'toggle-item';
            toggleDiv.innerHTML = `
                <input type="checkbox" id="${modelName}Toggle">
                <label for="${modelName}Toggle">${modelName}</label>
            `;
            togglesContainer.appendChild(toggleDiv);
            
            // Add event listener for toggle
            const toggle = toggleDiv.querySelector('input');
            toggle.addEventListener('change', (e) => {
                overlay.style.display = e.target.checked ? 'block' : 'none';
            });
            
            // Add metrics row
            const row = table.insertRow();
            row.innerHTML = `
                <td>${modelName}</td>
                <td>${(data.iou * 100).toFixed(2)}%</td>
            `;
        });
        
        metricsContainer.appendChild(table);
        
        // Make sure visualization container is visible
        document.getElementById('visualizationContainer').style.display = 'block';
        
        // Log for debugging
        console.log('Results displayed:', results);
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    window.modelInference = new ModelInference();
}); 