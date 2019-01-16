from flask import make_response

def create_response(response_string, status_code=200):
    response = make_response(response_string)
    response.headers['Access-Control-Allow-Origin'] = '*' #todo: set an appropriate value
    response.status_code = status_code
    return response