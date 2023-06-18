from flask import Flask, request, jsonify
from gpt import ask
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.post("/api/v1/chat-bot")
def detectSystemModel():
    question = request.form.get('question');
    # Encode the bytes using base64
    result_str = ask(question)

    return jsonify({'result': result_str})


if __name__ == '__main__':
    app.run()
