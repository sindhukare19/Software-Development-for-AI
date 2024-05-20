
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepow: car.Horsepow,
    }))
    .filter(car => (car.mpg != null && car.horsepow != null));
    return cleaned;
  }

  function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    mod.add(tf.layers.dense({units: 1, useBias: true}));
  
    return mod;
  }
  

  async function run() {
    const data = await getData();
    const values = data.map(d => ({
      x: d.horsepow,
      y: d.mpg,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Horsepow v MPG'},
      {values},
      {
        xLab: 'Horsepow',
        yLab: 'MPG',
        height: 300
      }
    );
  
    // More code will be added below
    const model = createModel();
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

     // Train the model
     await trainModel(model, inputs, labels);
      console.log('Done Training');
    testModel(model, data, tensorData);
  }
  
  document.addEventListener('DOMContentLoaded', run);
  
function convertToTensor(data) {
    
    return tf.tidy(() => {
      
      tf.util.shuffle(data);
  
      const inputs = data.map(d => d.horsepow)
      const labels = data.map(d => d.mpg);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }
  async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
  
    const batchSize = 32;
    const epochs = 55;
  
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }
  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
    const [xs, preds] = tf.tidy(() => {
  
      const xsNorm = tf.linspace(0, 1, 100);
      const predictions = model.predict(xsNorm.reshape([100, 1]));
  
      const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.horsepow, y: d.mpg,
    }));
  
  
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'},
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLab: 'Horsepow',
        yLab: 'MPG',
        height: 300
      }
    );
  }