import os
import pickle
import json 

from flask import Flask, jsonify, send_from_directory, abort, request

app = Flask(__name__)

if not os.path.exists('io'):
    os.makedirs('io')

@app.route('/')
def home():
    return 'Voice Transformation with VPC2020 baseline'


@app.route("/vpc", methods=["POST"])
def apply_vpc_baseline():
    """Upload an audio file, apply VPC on it and send the result."""
    params = json.loads(request.args.items().__next__()[0])
    # 1 Upload an audio file
    source_filepath = upload_audio()
 
    # 2 apply vpc
    result_filepath = '../io/output.wav'
    from subprocess import check_call
    cmd_vpc = ['./vpc/transform.sh']
    cmd_vpc.extend(['--anon-pool', 'anon_pool/train_other_500'])
    for key, value in params.items():
        cmd_vpc.extend(['--' + key, str(value)])
    cmd_vpc.extend(['../' + source_filepath, result_filepath])
    check_call(cmd_vpc)

    # 3 send the result
    return send_from_directory('io', 'output.wav', as_attachment=True)

def upload_audio():
    source_filepath = 'io/input.wav'
    with open(source_filepath, "wb") as fp:
        fp.write(request.data)
    return source_filepath

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

