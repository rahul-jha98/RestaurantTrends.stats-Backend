 
from flask import Flask,  request, render_template,Response
import requests
import json


app = Flask(__name__)


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

@app.route('/api/process')
def api():
    args = request.args
    
    if 'city' in args:
        city = args['city']
        
        city_name = search_city(city)
        if city != None:
            return Response(json.dumps({"status":1, "city_name":city_name}))
        else:
            return Response(json.dumps({"status":0}))
                            
    return Response(json.dumps({"status": 0}))

if __name__=='__main__':
    app.run()
