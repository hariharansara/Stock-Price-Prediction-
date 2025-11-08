// app.js — improved layout: title + legend positioned to avoid overlap, rangeslider visible
document.addEventListener('DOMContentLoaded', () => {
    // -------------------- config flags --------------------
    const USE_SMOOTH_PREDICTED = true;
    const SMOOTH_WINDOW = 3;
  
    const SHOW_RECENT_MARKERS = true;
    const RECENT_N = 12;
  
    // -------------------- helpers --------------------
    function movingAverage(arr, window = 3) {
      if (!Array.isArray(arr) || arr.length === 0) return [];
      const res = new Array(arr.length).fill(null);
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= window) {
          sum -= arr[i - window];
          res[i] = sum / window;
        } else {
          res[i] = sum / (i + 1);
        }
      }
      return res;
    }
  
    // -------------------- DOM --------------------
    const form = document.getElementById('predict-form');
    const statusDiv = document.getElementById('status');
    const metricsDiv = document.getElementById('metrics');
    const chartDiv = document.getElementById('chart');
  
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
  
      const ticker = document.getElementById('ticker').value.trim();
      const start = document.getElementById('start').value;
      const end = document.getElementById('end').value;
      const lookback = parseInt(document.getElementById('lookback').value);
      const epochs = parseInt(document.getElementById('epochs').value);
      const future_days = parseInt(document.getElementById('future_days').value || 0);
  
      statusDiv.textContent = 'Starting request — model may train for a while...';
      metricsDiv.textContent = '';
      chartDiv.innerHTML = '';
  
      try {
        const resp = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker, start, end, lookback, epochs, future_days })
        });
  
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          statusDiv.textContent = 'Error: ' + (err.error || resp.statusText || JSON.stringify(err));
          return;
        }
  
        const data = await resp.json();
        statusDiv.textContent = 'Prediction complete ✔';
        metricsDiv.textContent = `Ticker: ${data.ticker}  —  RMSE: ${Number(data.rmse).toFixed(2)}`;
  
        // Prepare arrays
        const dates = data.dates || [];
        const actual = (data.actual || []).map(Number);
        let predicted = (data.predicted || []).map(Number);
  
        // Optional smoothing for predicted
        if (USE_SMOOTH_PREDICTED && predicted.length > 0) {
          predicted = movingAverage(predicted, SMOOTH_WINDOW);
        }
  
        // Build traces
        const traces = [];
  
        // 1) Actual line
        traces.push({
          x: dates,
          y: actual,
          mode: 'lines',
          name: 'Actual (Test)',
          line: { width: 2, shape: 'spline' },
          hoverinfo: 'x+y'
        });
  
        // 2) Recent markers (visible & in legend)
        if (SHOW_RECENT_MARKERS && actual.length > 0) {
          const n = Math.min(RECENT_N, actual.length);
          const recentDates = dates.slice(-n);
          const recentActuals = actual.slice(-n);
          traces.push({
            x: recentDates,
            y: recentActuals,
            mode: 'markers',
            name: 'Recent (Markers)',
            showlegend: true,
            marker: { size: 7, opacity: 0.95, color: '#1f77b4' },
            hoverinfo: 'x+y'
          });
        }
  
        // 3) Predicted (test)
        traces.push({
          x: dates,
          y: predicted,
          mode: 'lines',
          name: 'Predicted (Test)',
          line: { dash: 'dash', width: 2, shape: 'spline' },
          hoverinfo: 'x+y'
        });
  
        // 4) Future predictions
        if (data.future_preds && data.future_dates) {
          const future_preds = (data.future_preds || []).map(Number);
          const future_dates = data.future_dates || [];
  
          traces.push({
            x: future_dates,
            y: future_preds,
            mode: 'lines+markers',
            name: `Predicted (Future ${future_dates.length} days)`,
            line: { width: 2, dash: 'dot', shape: 'spline' },
            marker: { symbol: 'triangle-up', size: 6, opacity: 0.95, color: '#059669' },
            hoverinfo: 'x+y'
          });
        }
  
        // Layout: explicit title + legend positions + rangeslider enabled
        const layout = {
          title: {
            text: `${data.ticker} — Actual vs Predicted (Test + Future)`,
            x: 0.5,
            xanchor: 'center',
            y: 0.98,
            yanchor: 'top',
            font: { size: 20 }
          },
          hovermode: 'x unified',
          xaxis: {
            title: 'Date',
            rangeslider: {
              visible: true,
              thickness: 0.08,
              bgcolor: 'rgba(255,255,255,0.6)'
            },
            rangeselector: {
              y: 0.96,
              buttons: [
                { count: 3, label: '3m', step: 'month', stepmode: 'backward' },
                { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
                { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
                { step: 'all' }
              ]
            },
            type: 'date'
          },
          yaxis: { title: 'Price', automargin: true },
          legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: 1.12,          // above plotting area but below card top
            yanchor: 'bottom',
            bgcolor: 'rgba(255,255,255,0.92)',
            bordercolor: 'rgba(0,0,0,0.06)',
            borderwidth: 1,
            font: { size: 12 }
          },
          margin: { t: 160, l: 60, r: 30, b: 100 }, // large top margin to leave room for title/legend/modebar
          height: 700,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(255,255,255,0.0)'
        };
  
        // Config: keep essential buttons; hide logo
        const config = {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['sendDataToCloud']
        };
  
        // Render
        Plotly.newPlot(chartDiv, traces, layout, config);
  
      } catch (err) {
        console.error(err);
        statusDiv.textContent = 'Unexpected error: ' + (err.message || err);
      }
    });
  });
  