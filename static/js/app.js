// static/js/app.js - polished plotting + toggles

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const statusDiv = document.getElementById('status');
  const metricsDiv = document.getElementById('metrics');
  const chartDiv = document.getElementById('chart');
  const btn = document.querySelector('.btn-gradient');

  const toggleRangeslider = document.getElementById('toggle_rangeslider');
  const toggleSmoothing = document.getElementById('toggle_smoothing');
  const toggleRecentMarkers = document.getElementById('toggle_recent_markers');

  // Simple moving average smoothing
  function movingAverage(arr, window = 3) {
    if (!Array.isArray(arr)) return [];
    const out = [];
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
      if (i >= window) {
        sum -= arr[i - window];
        out.push(sum / window);
      } else {
        out.push(sum / (i + 1));
      }
    }
    return out;
  }

  function setBusy(on) {
    if (on) {
      btn.setAttribute('disabled', 'disabled');
      btn.style.opacity = '0.7';
      statusDiv.textContent = 'Running model — may take a minute...';
    } else {
      btn.removeAttribute('disabled');
      btn.style.opacity = '';
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setBusy(true);
    chartDiv.innerHTML = '';
    metricsDiv.textContent = '';

    // gather inputs
    const payload = {
      ticker: document.getElementById('ticker').value.trim(),
      start: document.getElementById('start').value,
      end: document.getElementById('end').value,
      lookback: Number(document.getElementById('lookback').value || 60),
      epochs: Number(document.getElementById('epochs').value || 10),
      future_days: Number(document.getElementById('future_days').value || 30)
    };

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        statusDiv.textContent = 'Error: ' + (err.error || resp.statusText || JSON.stringify(err));
        setBusy(false);
        return;
      }

      const data = await resp.json();
      statusDiv.textContent = 'Prediction complete ✔';
      metricsDiv.textContent = `Ticker: ${data.ticker} — RMSE: ${Number(data.rmse).toFixed(2)}`;

      // prepare arrays
      const dates = data.dates || [];
      const actual = (data.actual || []).map(Number);
      let predicted = (data.predicted || []).map(Number);
      const future_preds = (data.future_preds || []).map(Number || []);
      const future_dates = data.future_dates || [];

      const useRangeslider = toggleRangeslider ? toggleRangeslider.checked : true;
      const useSmoothing = toggleSmoothing ? toggleSmoothing.checked : true;
      const showRecentMarkers = toggleRecentMarkers ? toggleRecentMarkers.checked : true;

      if (useSmoothing && predicted.length > 0) {
        predicted = movingAverage(predicted, 3);
      }

      // traces
      const traces = [];

      // Actual (lines, light markers off)
      traces.push({
        x: dates,
        y: actual,
        mode: 'lines',
        name: 'Actual (Test)',
        line: { width: 2.4, shape: 'spline' },
        hoverinfo: 'x+y'
      });

      // Recent markers (only last N)
      if (showRecentMarkers && actual.length > 0) {
        const N = Math.min(8, actual.length);
        traces.push({
          x: dates.slice(-N),
          y: actual.slice(-N),
          mode: 'markers',
          name: 'Recent (Markers)',
          marker: { size: 7, symbol: 'circle', color: '#1f77b4' },
          hoverinfo: 'x+y'
        });
      }

      // Predicted (test)
      traces.push({
        x: dates,
        y: predicted,
        mode: 'lines',
        name: 'Predicted (Test)',
        line: { dash: 'dash', width: 2.1, shape: 'spline' },
        hoverinfo: 'x+y'
      });

      // Future pred
      if (future_preds && future_preds.length > 0) {
        traces.push({
          x: future_dates,
          y: future_preds,
          mode: 'lines+markers',
          name: `Predicted (Future ${future_dates.length} days)`,
          line: { dash: 'dot', width: 2, shape: 'spline' },
          marker: { symbol: 'triangle-up', size: 6, color: '#059669' },
          hoverinfo: 'x+y'
        });
      }

      // layout: space for modebar + legend + title; legend centered in a neutral box
      const layout = {
        title: { text: `${data.ticker} — Actual vs Predicted (Test + Future)`, x: 0.5, font: { size: 20 } },
        hovermode: 'x unified',
        xaxis: {
          title: 'Date',
          rangeslider: { visible: !!useRangeslider, thickness: 0.06, bgcolor: 'rgba(240,240,240,0.85)' },
          rangeselector: {
            buttons: [
              { count: 3, label: '3m', step: 'month', stepmode: 'backward' },
              { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
              { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
              { step: 'all' }
            ]
          },
          type: 'date',
          tickfont: { size: 12 }
        },
        yaxis: { title: 'Price', automargin: true, tickfont: { size: 12 } },
        legend: {
          orientation: 'h',
          y: 1.08,
          x: 0.5,
          xanchor: 'center',
          bgcolor: 'rgba(255,255,255,0.92)',
          bordercolor: 'rgba(0,0,0,0.06)',
          borderwidth: 1,
          font: { size: 13 }
        },
        margin: { t: 130, l: 60, r: 30, b: 100 },
        height: 640,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(255,255,255,0.0)'
      };

      const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud'],
        toImageButtonOptions: { format: 'png', filename: `${data.ticker}_prediction` }
      };

      Plotly.newPlot(chartDiv, traces, layout, config);

    } catch (err) {
      console.error(err);
      statusDiv.textContent = 'Unexpected error: ' + (err.message || err);
    } finally {
      setBusy(false);
    }
  });
});
