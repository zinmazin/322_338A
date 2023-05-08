function predictDigit() {
  const imgData = canvas.toDataURL();
  $.ajax({
    type: "POST",
    url: "http://127.0.0.1:5000/predict",
    data: { img_data: imgData },
    success: function(result) {
      const labels = ['十一', '十二', '十三', '十四'];
      const prediction = labels[parseInt(result)];
      document.getElementById('result').innerHTML = 'Predicted digit: ' + prediction;
    },
    error: function(xhr, textStatus, errorThrown) {
      console.log('Error:', textStatus);
    }
  });
}
