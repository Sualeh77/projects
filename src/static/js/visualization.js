document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const visualizationContainer = document.getElementById('visualizationContainer');
    const baseImage = document.getElementById('baseImage');
    const maskOverlay = document.getElementById('maskOverlay');
    const opacitySlider = document.getElementById('opacitySlider');
    const brightnessSlider = document.getElementById('brightnessSlider');
    const contrastSlider = document.getElementById('contrastSlider');
    const gtMaskToggle = document.getElementById('gtMaskToggle');

    // Function to make mask background transparent
    function processGroundTruthMask(imageUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                
                // Draw the original mask
                ctx.drawImage(img, 0, 0);
                
                // Get the image data
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                
                // Process each pixel
                for (let i = 0; i < data.length; i += 4) {
                    // Check if pixel is black/background (assuming black is [0,0,0] or close to it)
                    const isBackground = data[i] < 10 && data[i + 1] < 10 && data[i + 2] < 10;
                    if (isBackground) {
                        // Make background pixels fully transparent
                        data[i + 3] = 0;
                    } else {
                        // Make building pixels white with current opacity
                        data[i] = 255;     // R
                        data[i + 1] = 255; // G
                        data[i + 2] = 255; // B
                        // Alpha channel remains unchanged
                    }
                }
                
                // Put the modified image data back
                ctx.putImageData(imageData, 0, 0);
                
                // Return the processed image URL
                resolve(canvas.toDataURL());
            };
            img.src = imageUrl;
        });
    }

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const imageFile = document.getElementById('imageUpload').files[0];
        const maskFile = document.getElementById('maskUpload').files[0];
        
        if (imageFile && maskFile) {
            // Load and display the image
            const imageReader = new FileReader();
            imageReader.onload = function(e) {
                baseImage.src = e.target.result;
                baseImage.onload = function() {
                    visualizationContainer.style.display = 'block';
                    updateImageEnhancement();
                }
            }
            imageReader.readAsDataURL(imageFile);
            
            // Load and process the mask
            const maskReader = new FileReader();
            maskReader.onload = function(e) {
                // Process the mask to make background transparent
                processGroundTruthMask(e.target.result).then(processedMaskUrl => {
                    maskOverlay.src = processedMaskUrl;
                    maskOverlay.style.opacity = opacitySlider.value / 100;
                });
            }
            maskReader.readAsDataURL(maskFile);
        }
    });

    opacitySlider.addEventListener('input', function(e) {
        maskOverlay.style.opacity = e.target.value / 100;
    });

    // Add ground truth toggle functionality
    gtMaskToggle.addEventListener('change', function(e) {
        maskOverlay.style.display = e.target.checked ? 'block' : 'none';
    });

    // Function to update image enhancement
    function updateImageEnhancement() {
        const brightness = brightnessSlider.value;
        const contrast = contrastSlider.value;
        baseImage.style.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    }

    // Add listeners for brightness and contrast controls
    brightnessSlider.addEventListener('input', updateImageEnhancement);
    contrastSlider.addEventListener('input', updateImageEnhancement);
}); 