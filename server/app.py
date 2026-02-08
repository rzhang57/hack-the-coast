import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv
from services.buffer_service import BufferService
from services.chat_service import ChatService

load_dotenv()
app = Flask(__name__)
CORS(app)

buffer_service = BufferService()
chat_service = ChatService()


@app.route('/key', methods=['GET'])
def has_api_key():
    userinfo_path = os.path.join(os.path.dirname(__file__), 'user_info.json')
    if os.path.exists(userinfo_path):
        with open(userinfo_path, 'r') as f:
            userinfo = json.load(f)
            if userinfo.get('gemini_api_key'):
                return jsonify({'hasKey': True})
    return jsonify({'hasKey': False})


@app.route('/key', methods=['POST'])
def set_api_key():
    api_key = request.json.get("api_key")
    if not api_key:
        return jsonify({'error': 'api_key is required'}), 400

    userinfo_path = os.path.join(os.path.dirname(__file__), 'user_info.json')
    userinfo = {}
    if os.path.exists(userinfo_path):
        with open(userinfo_path, 'r') as f:
            userinfo = json.load(f)

    userinfo['gemini_api_key'] = api_key
    with open(userinfo_path, 'w') as f:
        json.dump(userinfo, f)

    return jsonify({'hasKey': True})


@app.route('/buffer/record', methods=['POST'])
def start_context_buffer():
    return jsonify(buffer_service.start())


@app.route('/buffer/kill', methods=['POST'])
def kill_buffer():
    return jsonify(buffer_service.stop())


@app.route('/buffer/status', methods=['GET'])
def get_status():
    return jsonify(buffer_service.get_status())


def sse_stream(generator):
    def generate():
        for chunk in generator:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route('/assist', methods=['POST'])
def assist_request():
    snapshots = buffer_service.flush_buffer(clear=True)
    return sse_stream(chat_service.init_chat_stream(snapshots))


@app.route('/assist/chat', methods=['POST'])
def chat():
    message = request.json.get("message")
    return sse_stream(chat_service.send_message_stream(message))


@app.route('/')
def default():
    print("hello world")
