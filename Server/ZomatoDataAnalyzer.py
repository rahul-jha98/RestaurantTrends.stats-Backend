import json
import requests
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


##Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go


from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk



try:
    city_name = sys.srgv[1]
except Exception:
    city_name = 'Allahabad'




class Restaurant:
    def __init__(self, restaurant_json):
        self.restaurant_json = restaurant_json
        self.prepare_prameters_from_json()
    
    
    def get(self, value, json = ''):
        if json == '':
            json = self.restaurant_json
            
        if json == None:
            return None
        
        return json.get(value, None)

    def prepare_prameters_from_json(self):
        self.id = self.get('id')
        self.name = self.get('name')
        self.url = self.get('url')
        location = self.get('location')
        self.address = self.get('address', location)
        self.latitude = self.get('latitude', location)
        self.longitude = self.get('longitude', location)
        self.location = self.get('locality', location)
        self.city = self.get('locality', location)        
        self.online_order = 'Yes' if self.get('has_online_delivery') == 1 else 'No'
        self.book_table = 'Yes' if self.get('has_table_booking') == 1 else 'No'
        rating = self.get('aggregate_rating', self.get('user_rating'))
        if rating == None or rating == "NEW":
            self.rate = rating
        else:
            self.rate = '{}/5'.format(rating)
        self.votes = self.get('votes', self.get('user_rating'))
        phone = self.get('phone_numbers')
        if sum(c.isdigit() for c in phone) < 10:
            self.phone = None
        else:
            self.phone = phone
        self.rest_type = ', '.join(self.get('establishment'))
        self.cuisines = self.get('cuisines')
        self.approx_cost = self.get('average_cost_for_two')
        self.dish_liked = ''       
        self.reviews = []
  
    def set_reviews(self, reviews):
        self.reviews = reviews

    def set_dish_liked(self, dish_liked):
        self.dish_liked = dish_liked

    def get_row(self):
      
      return [self.url, self.address, self.name, self.online_order, self.book_table, self.rate, self.votes,
              self.phone, self.location, self.rest_type, self.dish_liked, self.cuisines, self.approx_cost, 
              self.reviews, self.latitude, self.longitude, self.location]



class ZomatoDatasetCreator:
    def __init__(self, city_name):
        
        # self.API_KEYS = ["1c1827e986cbb720c34bc661fdbd8884", 
        #                  "765fdb97e275ccf353c49c3c2ec68a7b",
        #                  "151799c34aa8943e8028a167e43f9588"]

        ## Abhi naya bana ke do daal do isme kal purana wala bhi append kar dena list me
        ## Basically ek city ke liye around 1500 calls hote hai so 
        self.API_KEYS = ["014af0114a43afec41812542b307726b", 
                        "1c1827e986cbb720c34bc661fdbd8884", 
                          "765fdb97e275ccf353c49c3c2ec68a7b",
                          "151799c34aa8943e8028a167e43f9588"
                         ]               
        self.BASE_URL = "https://developers.zomato.com/api/v2.1/"
        
        self.api_count = 0
        self.api_len = len(self.API_KEYS)
        
        self.city_name = city_name
    
    def get_response(self, end_point, query_params_dict):    
        url = self.BASE_URL + end_point
        
        self.api_count = (self.api_count + 1) % self.api_len
        
        return requests.get(url, 
                    params = query_params_dict,
                    headers = {'user-key':self.API_KEYS[self.api_count]}) 
    
    
    def search_city(self):
        response = self.get_response('locations', {'query': self.city_name})

        if response:
            result = response.json()['location_suggestions']

            if len(result) == 0:
                raise Exception("Search result is empty.")
            else:

                self.city_name = result[0]['city_name']
                self.city_id = result[0]['city_id']
                print("Setting the city name to ", self.city_name)
        else:
            raise Exception("Network Error")
            

    def fetch_establishments_dictionary(self):
        response = self.get_response('establishments', {'city_id': self.city_id})

        all_establishments = response.json()['establishments']

        estabishment_dict = {}

        for establishment in all_establishments:
            establishment = establishment['establishment']
            key, value = establishment.values()

            if type(value) == str:
                estabishment_dict[key] = value
            else:
                establishment_dict[value] = key

        return estabishment_dict
    
    
    def fetch_all_restaurants(self):
        
        self.all_restaurant_ids = {}

        establishments = self.fetch_establishments_dictionary()

        self.all_restaurants = []
        for sort_param in ['cost', 'rating']:
            self.sort_param = sort_param
            print('\nFetching restaurants by ' + self.sort_param)
            for e_id in tqdm(establishments.keys()):
                self.fetch_restaurants_of_type(e_id)  
            print('Total restaurants queried ' + str(len(self.all_restaurants)) + '\n')
                
 
    def fetch_restaurants_of_type(self, establishment_id, offset = 0, 
                                sort_order = 'desc', count = 20, max_pages = 5):
    
        if offset == max_pages:
            return

        response = self.get_response('search', {'entity_id': self.city_id, 'entity_type': 'city',
                                            'establishment_type': establishment_id,
                                           ' sort': self.sort_param,
                                            'order':sort_order,
                                            'start': offset * 20,
                                            'count':count})

        if response:
            result = response.json()
            queried = result['results_start'] + result['results_shown']
            total = result['results_found']
            count -= result['results_shown']


            restaurants_json = result['restaurants']

            for restaurant in restaurants_json:
                if self.all_restaurant_ids.get(restaurant['restaurant']['id'], -1) == -1:
                    self.all_restaurant_ids[restaurant['restaurant']['id']] = 0
                    self.all_restaurants.append(Restaurant(restaurant['restaurant']))


            if total > queried and sort_order == 'desc': 
                if offset + 1 < max_pages:
                    self.fetch_restaurants_of_type(establishment_id, offset + 1)
                else:
                    left = total - queried
                    self.fetch_restaurants_of_type(establishment_id, 0, 'asc', left)

            elif total > queried: 
                self.fetch_restaurants_of_type(establishment_id, offset + 1, 'asc', count)
            
        else:
            return None
        
 
    def fetch_reviews(self, res_id):
        response = self.get_response('reviews', 
                         {'res_id': res_id})
        
        reviews = []
        if response:
            response = response.json()
            fetched_reviews = response['user_reviews']
            
            for review in fetched_reviews:
                rating = 'Rated {}'.format(review['review']['rating'])
                text = review['review']['review_text']
                reviews.append((rating, text))
            
            return reviews
        else:
            return []
        
    
    def populate_reviews(self):
        for restaurant in tqdm(self.all_restaurants):
            restaurant.set_reviews(self.fetch_reviews(restaurant.id))
            
    def populate_dish_liked(self):
        
        def find_nth_occurance(string, char, n):
            val = -1
            for i in range(n):
                val = string.find(char, val + 1)
            return val
        

        for restaurant in tqdm(self.all_restaurants):
            
            URL = restaurant.url

            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
            response = requests.get(URL.split('?')[0],headers=headers)     

            html = response.text

            pos = html.find('Top Dishes People Order')
            if  pos!= -1:
                remaining = html[pos:]
                start = find_nth_occurance(remaining, '>', 2)
                end = find_nth_occurance(remaining, '<', 3)
                restaurant.set_dish_liked(remaining[start + 1:end])



data_creator = ZomatoDatasetCreator(city_name)
data_creator.search_city()
data_creator.fetch_all_restaurants()
data_creator.populate_reviews()
data_creator.populate_dish_liked()
data_list = data_creator.all_restaurants

np_data  = np.empty((len(data_list), len(data_list[0].get_row())), dtype=object)
for i in tqdm(range(len(data_list[:]))):
  rest = data_list[i]
  np_data[i] = rest.get_row()


df = pd.DataFrame(np_data, columns = ['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes', 'phone', 'location',
                                      'rest_type', 'dish_liked', 'cuisines', 'approx_cost', 'reviews', 'latitude', 'longitude', 'location'])


os.mkdir(city_name)
df.to_csv(city_name+'/data.csv')


