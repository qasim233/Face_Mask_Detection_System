// Get DOM elements
const webcamButton = document.getElementById("webcamButton")
const uploadButton = document.getElementById("uploadButton")
const fileInput = document.getElementById("fileInput")
const webcamContainer = document.getElementById("webcamContainer")
const uploadPreview = document.getElementById("uploadPreview")
const uploadedImage = document.getElementById("uploadedImage")
const closeUpload = document.getElementById("closeUpload")
const toggleButton = document.getElementById("toggleButton")
const detectButton = document.getElementById("detectButton")
const statusIndicator = document.getElementById("statusIndicator")
const webcamFeed = document.getElementById("webcamFeed")
const processingOverlay = document.getElementById("processingOverlay")
const processingText = document.getElementById("processingText")
const progressBar = document.getElementById("progressBar")

// Stats elements
const totalFaces = document.getElementById("totalFaces")
const maskedFaces = document.getElementById("maskedFaces")
const unmaskedFaces = document.getElementById("unmaskedFaces")
const confidence = document.getElementById("confidence")

// State variables
let webcamActive = true
let currentMode = "webcam" // 'webcam' or 'upload'

// Initialize UI
function initUI() {
  // Set initial state
  webcamContainer.style.display = "block"
  uploadPreview.style.display = "none"
  toggleButton.style.display = "inline-block"
  detectButton.style.display = "none"

  // Reset stats
  updateStats(0, 0, 0, 0)
}

// Switch to webcam mode
function switchToWebcam() {
  currentMode = "webcam"
  webcamContainer.style.display = "block"
  uploadPreview.style.display = "none"
  toggleButton.style.display = "inline-block"
  detectButton.style.display = "none"

  // Restart webcam if it was active
  if (webcamActive) {
    webcamFeed.src = "/video"
    statusIndicator.classList.remove("inactive")
    webcamContainer.classList.remove("inactive")
  }

  // Reset stats
  updateStats(0, 0, 0, 0)
}

// Switch to upload mode
function switchToUpload() {
  currentMode = "upload"
  fileInput.click()
}

// Handle file selection
function handleFileSelect(event) {
  const file = event.target.files[0]
  if (!file) return

  // Check if file is an image
  if (!file.type.match("image.*")) {
    alert("Please select an image file.")
    return
  }

  // Create object URL for the image
  const imageUrl = URL.createObjectURL(file)

  // Set image source directly
  uploadedImage.src = imageUrl

  // Make the image fill the container better
  uploadedImage.style.width = "auto"
  uploadedImage.style.height = "auto"
  uploadedImage.style.maxWidth = "100%"
  uploadedImage.style.maxHeight = "100%"
  uploadedImage.style.objectFit = "contain"

  // Show upload preview, hide webcam
  webcamContainer.style.display = "none"
  uploadPreview.style.display = "flex"
  toggleButton.style.display = "none"
  detectButton.style.display = "inline-block"

  // Stop webcam feed if active
  if (webcamActive) {
    webcamFeed.src = ""
  }

  // Reset stats
  updateStats(0, 0, 0, 0)
}

// Close upload preview
function closeUploadPreview() {
  // Clear the file input
  fileInput.value = ""

  // Switch back to webcam mode
  switchToWebcam()
}

// Toggle webcam feed
function toggleFeed() {
  webcamActive = !webcamActive

  if (webcamActive) {
    // Resume feed
    webcamFeed.src = "/video"
    toggleButton.textContent = "Stop Feed"
    toggleButton.classList.remove("stop")
    webcamContainer.classList.remove("inactive")
    statusIndicator.classList.remove("inactive")
  } else {
    // Stop feed
    webcamFeed.src = ""
    toggleButton.textContent = "Start Feed"
    toggleButton.classList.add("stop")
    webcamContainer.classList.add("inactive")
    statusIndicator.classList.add("inactive")
  }
}

// Detect masks in uploaded image
async function detectMasks() {
  // Show processing overlay
  showProcessingOverlay("ANALYZING")

  // Grab the uploaded file
  const file = fileInput.files[0]
  if (!file) {
    alert("Please upload an image first.")
    hideProcessingOverlay()
    return
  }

  // Create form data for the file
  const formData = new FormData()
  formData.append("file", file)

  try {
    // Simulate progress
    simulateProcessing(2000)

    // Send to Flask endpoint
    const response = await fetch("/detect", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`)
    }

    // Get the response data as JSON
    const data = await response.json()

    // Update the image with the processed result
    uploadedImage.src = data.image_base64

    // Update stats with the actual values from the backend
    updateStats(data.total_faces, data.masked_faces, data.unmasked_faces, data.confidence)
  } catch (error) {
    console.error("Error detecting masks:", error)
    alert("Error processing image. Please try again.")
  } finally {
    // Hide processing overlay
    hideProcessingOverlay()
  }
}

// Show processing overlay
function showProcessingOverlay(text) {
  processingText.textContent = text
  processingOverlay.style.display = "flex"
  progressBar.style.width = "0%"
}

// Hide processing overlay
function hideProcessingOverlay() {
  processingOverlay.style.display = "none"
}

// Simulate processing with progress
function simulateProcessing(duration) {
  const startTime = Date.now()
  const interval = 30 // Update interval in ms

  function updateProgress() {
    const elapsed = Date.now() - startTime
    const progress = Math.min((elapsed / duration) * 100, 100)

    progressBar.style.width = `${progress}%`

    if (progress < 100) {
      setTimeout(updateProgress, interval)
    }
  }

  updateProgress()
}

// Update stats display
function updateStats(total, masked, unmasked, conf) {
  // Animate counting up
  animateCounter(totalFaces, 0, total, 1000)
  animateCounter(maskedFaces, 0, masked, 1000)
  animateCounter(unmaskedFaces, 0, unmasked, 1000)
  animateCounter(confidence, 0, conf, 1000, "%")
}

// Animate counter
function animateCounter(element, start, end, duration, suffix = "") {
  const range = end - start
  const startTime = Date.now()

  function updateCounter() {
    const elapsed = Date.now() - startTime
    const progress = Math.min(elapsed / duration, 1)

    const current = Math.floor(start + range * progress)
    element.textContent = current + suffix

    if (progress < 1) {
      requestAnimationFrame(updateCounter)
    }
  }

  updateCounter()
}

// Event listeners
webcamButton.addEventListener("click", switchToWebcam)
uploadButton.addEventListener("click", switchToUpload)
fileInput.addEventListener("change", handleFileSelect)
closeUpload.addEventListener("click", closeUploadPreview)
toggleButton.addEventListener("click", toggleFeed)
detectButton.addEventListener("click", detectMasks)

// Initialize on page load
window.addEventListener("DOMContentLoaded", initUI)
