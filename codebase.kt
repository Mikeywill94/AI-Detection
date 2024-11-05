// This is a suspend function that takes a Bitmap and returns a list of detected object labels.
private suspend fun preformObjectDetection(bitmap: Bitmap): List<String> {

    // Use coroutine context to perform the inference on a background thread (I/O dispatcher).
    return withContext(Dispatchers.IO) {

        // Load the image data (byteBuffer) into the TensorBuffer input.
        inputFeature0.loadBuffer(byteBuffer)

        // Run the YOLO model on the input image and get the output (predictions).
        val outputs = model.process(inputFeature0)

        // Retrieve the output tensor buffer from the model (contains scores, bounding boxes, etc.).
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Initialize an empty list to hold the predicted object labels.
        val predictions = mutableListOf<String>()

        // Get the array of scores from the model's output.
        val scores = outputFeature0.floatArray

        // Define the number of detections the model will return (6300 in this case).
        val numDetections = 6300 // Based on the model's output format

        // Predefined labels for the YOLO model (80 object classes).
        val labelNames = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
            "toothbrush"
        )

        // Loop through all detections to find objects with confidence scores higher than the threshold.
        for (i in 0 until numDetections) {
            // Extract the confidence scores for each class for the current detection.
            val classScores = scores.copyOfRange(i * 85 + 5, i * 85 + 85)

            // Find the index of the class with the highest confidence score.
            val maxScoreIndex = classScores.indices.maxByOrNull { classScores[it] } ?: -1

            // Get the actual confidence score for that class.
            val maxScore = if (maxScoreIndex >= 0) classScores[maxScoreIndex] else 0f

            // If the confidence score is above the threshold, add the corresponding label to predictions.
            if (maxScore >= confidenceThreshold) {
                if (maxScoreIndex < labelNames.size) {
                    predictions.add(labelNames[maxScoreIndex])
                }
            }
        }

        // Close the model instance to release resources.
        model.close()

        // Return the list of predictions (detected objects).
        predictions
    }
}
