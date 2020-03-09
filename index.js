// Notice there is no 'import' statement. 'mobilenet' and 'tf' is
// available on the index-page because of the script tag above.

// Load the model.
(async () => {

  tf.setBackend('webgl');
  const model = await tf.loadGraphModel('quantized - lite/model.json');
  console.log('model loaded: ', model)
  const dummy =tf.zeros([1, 300,300,3])
  const dummyPrediction = await model.executeAsync({ image_tensor: dummy })
  dummy.dispose();
  for(let i = 0; i < dummyPrediction.length; i++) {
    dummyPrediction[i].dispose();
  }

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const status = document.getElementById("status");
  const context = canvas.getContext("2d");
  canvas.width = document.body.clientWidth; //document.width is obsolete
  canvas.height = document.body.clientHeight; //document.height is obsolete
  video.width = document.body.clientWidth; //document.width is obsolete
  video.height = document.body.clientHeight; //document.height is obsolete
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "environment"
    }
  });

  video.srcObject = stream;
  video.onloadeddata = function() {
    detect();
  }

  
  async function detect() {
    
    const tfImg = await tf.browser.fromPixels(video)
    const smallImg = tf.image.resizeBilinear(tfImg, [300, 300]) // 600, 450
    const tf4d = smallImg.expandDims()
    // const resized = tf.cast(smallImg, tf.float16)
    // const tf4d = tf.tensor4d(Array.from(resized.dataSync()), [1, 300, 300, 3]) // 600, 450
    const predictions = await model.executeAsync({ image_tensor: tf4d }, ['detection_boxes', 'num_detections', 'detection_classes', 'detection_scores'])
    tf4d.dispose();
    smallImg.dispose();
    tfImg.dispose();

    
    const predictionBoxes = predictions[0].dataSync();
    const predictionScores = predictions[3].dataSync();
    // console.log(predictionScores>0.3)
    // console.log("Test")
    context.clearRect(0, 0, canvas.width, canvas.height);   

    // const result = await model.predict(video);

    context.beginPath();
    for (let i = 0; i < 5; i++) {
        const score = predictionScores[i] * 100

        // console.log(score)
        
      if (score > 30) {
            const minY = predictionBoxes[i * 4] * video.height
            const minX = predictionBoxes[i * 4 + 1] * video.width
            const maxY = predictionBoxes[i * 4 + 2] * video.height
            const maxX = predictionBoxes[i * 4 + 3] * video.width
            
            // console.log('model loaded: ', minX, minY, maxX - minX, maxY - minY)
            
            context.rect(minX, minY, maxX - minX, maxY - minY);
            // context.rect(100, 100, 100,100);

            context.lineWidth = canvas.width / 100;
            context.strokeStyle = "cyan";
            context.fillStyle = "cyan";
            context.font = "" + canvas.height / 25 + "px Arial";
            context.stroke();
            context.fillText(
            "Shoe - " +  score.toFixed(1),
            minX,
            minY > 10 ? minY - 5 : 10
            );
      }
    }
    predictions[0].dispose();
    predictions[1].dispose();
    predictions[2].dispose();
    predictions[3].dispose();

    requestAnimationFrame(detect);
  }
})();
