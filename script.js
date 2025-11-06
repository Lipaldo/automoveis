const chartCtx = document.getElementById('chart').getContext('2d');
const trainStatus = document.getElementById('train-status');
const maeSpan = document.getElementById('mae');
const tempoSpan = document.getElementById('tempo');

const chart = new Chart(chartCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Erro (MAE)',
      data: [],
      borderColor: '#00bcd4',
      borderWidth: 2,
      fill: false
    }]
  },
  options: {
    scales: {
      x: { title: { display: true, text: 'Época' }},
      y: { title: { display: true, text: 'Erro (MAE)' }}
    }
  }
});

async function criarRede() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 158, activation: 'relu', inputShape: [316] }));
  model.add(tf.layers.dense({ units: 158, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanAbsoluteError',
    metrics: ['mae']
  });

  return model;
}

document.getElementById('train-btn').addEventListener('click', async () => {
  trainStatus.textContent = "Treinando rede neural...";
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update();

  const inicio = performance.now();
  const model = await criarRede();

  // Gera dados sintéticos (simula base real)
  const X = tf.randomNormal([1000, 316]);
  const y = tf.randomNormal([1000, 1]);

  await model.fit(X, y, {
    epochs: 50,
    batchSize: 64,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        chart.data.labels.push(epoch + 1);
        chart.data.datasets[0].data.push(logs.mae.toFixed(4));
        chart.update();
        trainStatus.textContent = `Época ${epoch + 1}/50 - Erro: ${logs.mae.toFixed(4)}`;
      }
    }
  });

  const fim = performance.now();
  const tempo = ((fim - inicio) / 1000).toFixed(2);
  const finalMae = chart.data.datasets[0].data.slice(-1)[0];

  maeSpan.textContent = finalMae;
  tempoSpan.textContent = tempo + " s";
  trainStatus.textContent = "✅ Treinamento concluído!";
});
