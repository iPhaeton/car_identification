from flask import Flask, jsonify, request

app = Flask(__name__)


def load_model():
    '''Load and return the model'''
    # TODO: INSERT CODE
    # return model

    return {}


# you can then reference this model object in evaluate function/handler
model = load_model()


# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    input_image = request.files.get('image')

    return "success"


# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)