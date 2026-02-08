from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from services.buffer_service import BufferService
from services.chat_service import ChatService

load_dotenv()
app = Flask(__name__)
CORS(app)

buffer_service = BufferService()
chat_service = ChatService()


@app.route('/buffer/record', methods=['POST'])
def start_context_buffer():
    return jsonify(buffer_service.start())


@app.route('/buffer/kill', methods=['POST'])
def kill_buffer():
    return jsonify(buffer_service.stop())


@app.route('/buffer/status', methods=['GET'])
def get_status():
    return jsonify(buffer_service.get_status())


@app.route('/assist', methods=['POST'])
def assist_request():
    snapshots = buffer_service.flush_buffer(clear=True)
    response_text = chat_service.init_chat(snapshots)
    return jsonify({"response": response_text})


@app.route('/assist/chat', methods=['POST'])
def chat():
    message = request.json.get("message")
    response_text = chat_service.send_message(message)
    return jsonify({"response": response_text})


@app.route('/')
def default():
    print("hello world")
