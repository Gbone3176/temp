from flask import Flask, request, jsonify
from flasgger import Swagger
import random

app = Flask(__name__)
Swagger(app)


@app.route('/api/<string:language>/', methods=['GET'])
def language_awesomeness(language):
    """
    This is the language awesomeness API
    Call this api passing a language name and get back its features
    ---
    tags:
      - Awesomeness Language API
    parameters:
      - name: language
        in: path
        type: string
        required: true
        description: The language name
      - name: size
        in: query
        type: integer
        description: size of awesomeness
    responses:
      500:
        description: Error The language is not awesome!
      200:
        description: A language with its awesomeness
        schema:
          id: awesome
          properties:
            language:
              type: string
              description: The language name
              default: Lua
            features:
              type: array
              description: The awesomeness list
              items:
                type: string
              default: ["perfect", "simple", "lovely"]
    """

    language = language.lower().strip()
    features = [
        "awesome", "great", "dynamic",
        "simple", "powerful", "amazing",
        "perfect", "beauty", "lovely"
    ]
    size = int(request.args.get('size', 1))
    if language in ['php', 'vb', 'visualbasic', 'actionscript']:
        return "An error occurred, invalid language for awesomeness", 500
    return jsonify(
        language=language,
        features=random.sample(features, size)
    )


@app.route('/api/random_number/', methods=['GET'])
def random_number():
    """
    Get a random number
    ---
    tags:
      - Random Number API
    responses:
      200:
        description: A random number
    """
    return jsonify(number=random.randint(1, 100))


@app.route('/api/greet/<string:name>/', methods=['GET'])
def greet_user(name):
    """
    Greet the user
    ---
    tags:
      - Greeting API
    parameters:
      - name: name
        in: path
        type: string
        required: true
        description: The name of the person to greet
    responses:
      200:
        description: A greeting to the user
    """
    return jsonify(greeting=f"Hello, {name}!")


@app.route('/api/echo/', methods=['POST'])
def echo():
    """
    Echo the message sent in POST data
    ---
    tags:
      - Echo API
    parameters:
      - in: body
        name: body
        required: true
        description: Message to echo
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: The message to echo
              example: "Hello, World!"
    responses:
      200:
        description: The message sent by the user
    """
    data = request.json
    return jsonify(echo=data.get('message', ''))


@app.route('/api/calculate/', methods=['GET'])
def calculate():
    """
    A simple calculator
    ---
    tags:
      - Calculator API
    parameters:
      - name: operation
        in: query
        type: string
        required: true
        enum: ['add', 'subtract', 'multiply', 'divide']
        description: The operation to perform
      - name: x
        in: query
        type: number
        required: true
        description: The first number
      - name: y
        in: query
        type: number
        required: true
        description: The second number
    responses:
      200:
        description: The result of the calculation
      400:
        description: Invalid input
    """
    operation = request.args.get('operation')
    x = request.args.get('x', type=float)
    y = request.args.get('y', type=float)
    result = None

    if operation == 'add':
        result = x + y
    elif operation == 'subtract':
        result = x - y
    elif operation == 'multiply':
        result = x * y
    elif operation == 'divide':
        if y != 0:
            result = x / y
        else:
            return jsonify(error="Cannot divide by zero."), 400
    else:
        return jsonify(error="Invalid operation."), 400

    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)

