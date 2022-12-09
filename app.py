from predict import Predict
from flask import Flask, request
from flask_cors import cross_origin, CORS

app = Flask(__name__)
CORS(app)
predict = Predict()
print("Ready!")

@app.route("/coreference", methods=["POST"])
@cross_origin()
def post_classify_review():
    if request.is_json:
        try:
            text = request.get_json()["coreftext"]
        except (KeyError):
            return {"error":"Request must contain a text"}, 400
        label = predict.predict(text)
        return { "coreference" : label}, 200
    return {"error":"Request must be JSON"}, 415
