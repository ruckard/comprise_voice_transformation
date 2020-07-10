import os
import pickle
import json 

from flask import Flask, jsonify, send_from_directory, abort, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'Voice Transformation with VPC2020 baseline'


@app.route("/vpc", methods=["POST"])
def apply_vpc_baseline():
    """Upload an audio file, apply VPC on it and send the result."""
    params = json.loads(request.args.items().__next__()[0])
    # 1 Upload an audio file
    source_filepath = upload_audio()
    # 2 create params file from request
    with open('config_transform.def.sh', mode='r') as default_values, open('config_transform.sh', mode='w') as f:
        for line in default_values:
            if line.strip() and line[0] != '#':
                name, value = line.split('=')
                name = name.strip()
                if name in params:
                    value = '"{}"'.format(params[name])
                f.write('{}={}\n'.format(name, value))

 
    # 3 apply vpc
    result_filepath = 'io/results/output.wav'
    from subprocess import check_call
    cmd_vpc = ['./transform.sh', '--ipath',  source_filepath, '--opath', result_filepath]
    check_call(cmd_vpc)

    # 4 send the result
    return send_from_directory('io/results', 'output.wav', as_attachment=True)

def upload_audio():
    source_filepath = 'io/inputs/input.wav'
    with open(source_filepath, "wb") as fp:
        fp.write(request.data)
    return source_filepath

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

