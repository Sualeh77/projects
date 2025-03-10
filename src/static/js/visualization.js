document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const visualizationContainer = document.getElementById('visualizationContainer');
    const baseImage = document.getElementById('baseImage');
    const maskOverlay = document.getElementById('maskOverlay');
    const opacitySlider = document.getElementById('opacitySlider');
    const brightnessSlider = document.getElementById('brightnessSlider');
    const contrastSlider = document.getElementById('contrastSlider');

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
            
            // Load and display the mask
            const maskReader = new FileReader();
            maskReader.onload = function(e) {
                maskOverlay.src = e.target.result;
                maskOverlay.style.opacity = opacitySlider.value / 100;
            }
            maskReader.readAsDataURL(maskFile);
        }
    });

    opacitySlider.addEventListener('input', function(e) {
        maskOverlay.style.opacity = e.target.value / 100;
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