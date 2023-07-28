from flask import Flask, jsonify, render_template

from modeling import modeling

app = Flask(__name__)

# Reload template if cache is invalidated
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


# API
@app.route("/api/models_info.json")
def api_models_info():
    return jsonify([m.structure_json() for m in modeling.loaded_models])


if __name__ == "__main__":
    modeling.load_model("facebook/opt-125m")
    app.run()
