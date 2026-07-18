# api_comparison.py

# Flask Implementation
from flask import Flask

flask_app = Flask(__name__)

@flask_app.route('/hello', methods=['GET'])
def flask_hello():
    return {'message': 'Hello from Flask!', 'framework': 'Flask'}

# FastAPI Implementation
from fastapi import FastAPI

fastapi_app = FastAPI()

@fastapi_app.get("/hello")
def fastapi_hello():
    return {"message": "Hello from FastAPI!", "framework": "FastAPI"}

if __name__ == "__main__":
    print("Flask app defined at /hello")
    print("FastAPI app defined at /hello")
