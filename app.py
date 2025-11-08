import os
import math
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.metrics import mean_squared_error

# Import train/load helpers from model_utils (must be in same folder)
from model_utils import train_model_for_ticker, fetch_stock, model_filepath_for_ticker

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = True if os.environ.get("FLASK_DEBUG", "1") == "1" else False

# ---------- App init ----------
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-lstm-app")

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    """Render the main dashboard page (templates/index.html)."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "ticker": "AAPL",
      "start": "2018-01-01",
      "end": "2024-01-01",
      "lookback": 60,
      "epochs": 10,
      "batch_size": 32,        # optional
      "future_days": 30,       # optional
      "force_retrain": false   # optional
    }
    """
    try:
        payload = request.get_json(force=True)
        ticker = payload.get("ticker", "AAPL")
        if ticker:
            ticker = ticker.upper().strip()
        start = payload.get("start")
        end = payload.get("end")
        lookback = int(payload.get("lookback", 60))
        epochs = int(payload.get("epochs", 10))
        batch_size = int(payload.get("batch_size", 32))
        future_days = int(payload.get("future_days", 0))
        force_retrain = bool(payload.get("force_retrain", False))

        # Basic validation
        if not start or not end:
            return jsonify({"error": "start and end dates required (YYYY-MM-DD)"}), 400

        # Validate we can fetch data for this ticker/date range
        try:
            df_check = fetch_stock(ticker, start, end)
            if df_check.empty or len(df_check) < (lookback + 10):
                return jsonify({"error": "Not enough data for given lookback / date range."}), 400
        except Exception as e:
            logger.exception("Failed to fetch data for validation")
            return jsonify({"error": f"Failed to fetch data for {ticker}: {str(e)}"}), 400

        logger.info(
            "Predict request: ticker=%s start=%s end=%s lookback=%d epochs=%d future_days=%d force_retrain=%s",
            ticker, start, end, lookback, epochs, future_days, force_retrain
        )

        # Train (or load saved) model and produce predictions
        result = train_model_for_ticker(
            ticker=ticker,
            start=start,
            end=end,
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size,
            future_days=future_days,
            force_retrain=force_retrain,
        )

        preds = result.get("preds")
        actuals = result.get("actuals")
        dates = result.get("dates")

        if preds is None or actuals is None or dates is None:
            return jsonify({"error": "Model failed to return predictions."}), 500

        rmse = math.sqrt(mean_squared_error(actuals, preds))

        resp = {
            "ticker": ticker,
            "dates": dates,
            "predicted": [float(x) for x in preds.tolist()],
            "actual": [float(x) for x in actuals.tolist()],
            "rmse": float(rmse),
        }

        if "future_preds" in result:
            resp["future_preds"] = [float(x) for x in result["future_preds"].tolist()]
            resp["future_dates"] = result["future_dates"]

        # Also include model path (useful for debugging; optional)
        try:
            model_path = model_filepath_for_ticker(ticker)
            if os.path.exists(model_path):
                resp["model_path"] = model_path
        except Exception:
            pass

        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unhandled exception in /api/predict")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def list_models():
    """
    Optional helper endpoint to list available saved models (models/*.h5).
    Returns list of {"ticker":..., "path":...}
    """
    try:
        from model_utils import MODELS_DIR
        models = []
        if os.path.exists(MODELS_DIR):
            for fname in os.listdir(MODELS_DIR):
                if fname.lower().endswith(".h5"):
                    ticker = os.path.splitext(fname)[0]
                    models.append({"ticker": ticker, "path": os.path.join(MODELS_DIR, fname)})
        return jsonify({"models": models}), 200
    except Exception as e:
        logger.exception("Error listing models")
        return jsonify({"error": "Failed to list models", "details": str(e)}), 500


@app.route("/api/delete_model", methods=["POST"])
def delete_model():
    """
    Optional admin endpoint to delete a saved model.
    Expects JSON: {"ticker": "AAPL"}
    """
    try:
        payload = request.get_json(force=True)
        ticker = payload.get("ticker", "")
        if not ticker:
            return jsonify({"error": "ticker required"}), 400
        from model_utils import model_filepath_for_ticker
        path = model_filepath_for_ticker(ticker)
        if os.path.exists(path):
            os.remove(path)
            return jsonify({"deleted": path}), 200
        else:
            return jsonify({"error": "model not found", "path": path}), 404
    except Exception as e:
        logger.exception("Error deleting model")
        return jsonify({"error": "Failed to delete model", "details": str(e)}), 500


# ---------- Run ----------
if __name__ == "__main__":
    logger.info("Starting Flask app on %s:%d (templates=%s, static=%s)", HOST, PORT, TEMPLATES_DIR, STATIC_DIR)
    # NOTE: debug=False and use_reloader=False prevents the Werkzeug watcher restarting
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
