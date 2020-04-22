 
from flask import Flask, jsonify, request, render_template,Response
import requests
import json
import os, sys
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

def get_response(end_point, query_params_dict):    
    url = "https://developers.zomato.com/api/v2.1/" + end_point
    
    return requests.get(url, 
                params = query_params_dict,
                headers = {'user-key':"151799c34aa8943e8028a167e43f9588"}) 
    
def search_city(city_name):
    response = get_response('locations', {'query': city_name})

    if response:
        result = response.json()['location_suggestions']

        if len(result) == 0:
            return None
        else:
            city_name = result[0]['city_name']
            return city_name
    else:
        raise Exception("Network Error")

def get_response_header(json):
    response = json
    return response

@app.route('/api/process')
def api():
    args = request.args
    print('Returning', args);
    if 'city' in args:
        city = args['city']
        print(city)
        city_name = search_city(city)
        if city != None:
            pid = os.fork()
            
            if pid > 0:
                return get_response_header(jsonify(status=1, city_name=city_name))
            else:
                os.execl(sys.executable, 'python3', "ZomatoDataAnalyzer.py", city_name, *sys.argv[1:])
                os._exit(0)
            
        else:
            return get_response_header(jsonify(status=0))
                            
    return get_response_header(jsonify(status=0))


@app.after_request
def add_header(response):
    response.headers['content-type'] = 'application/json; charset=utf-8'
    return response

if __name__=='__main__':
    app.run()
