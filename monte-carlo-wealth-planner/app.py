import base64

from flask import Flask, render_template, request

from monte_carlo import render_histogram_png, simulate_growth, summarize

app = Flask(__name__)

DEFAULT_MEAN_RETURN = 0.07
DEFAULT_VOLATILITY = 0.15
DEFAULT_SIMULATIONS = 10_000


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        initial = float(request.form["initial"])
        contribution = float(request.form["contribution"])
        years = int(request.form["years"])

        final_values = simulate_growth(
            initial_investment=initial,
            annual_contribution=contribution,
            years=years,
            mean_return=DEFAULT_MEAN_RETURN,
            volatility=DEFAULT_VOLATILITY,
            num_simulations=DEFAULT_SIMULATIONS,
        )

        stats = summarize(final_values)
        histogram_b64 = base64.b64encode(
            render_histogram_png(final_values, initial)
        ).decode("ascii")

        results = {
            "initial": initial,
            "contribution": contribution,
            "years": years,
            "median": stats["median"],
            "p10": stats["p10"],
            "p90": stats["p90"],
            "histogram_b64": histogram_b64,
        }

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
