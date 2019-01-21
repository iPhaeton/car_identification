from flask import make_response
import sys
sys.path.append("..")
from constants import allowed_origins

def create_response(request, response_string, status_code=200):
    response = make_response(response_string)
    if 'HTTP_ORIGIN' in request.environ and request.environ['HTTP_ORIGIN']  in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', request.environ['HTTP_ORIGIN'] )
        response.headers.add('Access-Control-Allow-Headers', 'access-control-allow-origin,content-type')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    response.status_code = status_code
    return response

def get_pretty_classnames(classnames, classnames_map):
    return list(map(lambda cl: classnames_map[cl], classnames))