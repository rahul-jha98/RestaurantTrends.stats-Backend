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




try:
    city_name = sys.argv[1]
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

df = pd.read_csv(city_name + '/data.csv');

fig_dat = {}

plt.figure(figsize=(10,10))
chains=df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
# plt.title("Most famous restaurants chains in Bengaluru")
plt.xlabel("Number of outlets")
plt.savefig(city_name+'/img1.png')
plt.close()
image_name = 'Number of Outlets'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Histogram displaying most famous restaurants chains in ' + city_name,
                       'path': 'img1.png', 'type':'image'}

x=df['online_order'].value_counts()
colors = ['#FEBFB3', '#E1396C']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="",width=500,height=500)
fig=go.Figure(data=[trace])
fig.write_html(city_name+"/html1.html", auto_open = False)

image_name = 'Order Type'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Pie Chart comparing online and offline orders',
                       'path': 'html1.html', 'type':'html'}


py.plot(fig, auto_open = False)


x=df['book_table'].value_counts()
colors = ['#96D38C', '#D0F9B1']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="Table booking",width=500,height=500)
fig=go.Figure(data=[trace])

fig.write_html(city_name+"/html2.html")

image_name = 'Booking Type'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Pie chart showing booking types',
                       'path': 'html2.html', 'type':'html'}

py.plot(fig, auto_open = False)

plt.figure(figsize=(10,10))
rating=df['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()
sns.distplot(rating,bins=20)
plt.savefig(city_name+'/img2.png')
plt.close()

image_name = 'Histogram of Ratings'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Histogram displaying most famous restaurants chains in ' + city_name,
                       'path': 'img2.png', 'type':'image'}

cost_dist=df[['rate','approx_cost','online_order']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

plt.figure(figsize=(10,10))
sns.scatterplot(x="rate",y='approx_cost',hue='online_order',data=cost_dist)
plt.savefig(city_name+'/img3.png')

plt.close()

image_name = 'Approx cost for two'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Approximate cost for two people in the restaurant',
                       'path': 'img3.png', 'type':'image'}


plt.figure(figsize=(10,10))
sns.distplot(cost_dist['approx_cost'])
plt.savefig(city_name+'/img4.png')

plt.close()

image_name = 'Histogram: Approx cost for two'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Histogram plot of approximate cost for two people in the restaurant',
                       'path': 'img4.png', 'type':'image'}


votes_yes=df[df['online_order']=="Yes"]['votes']
trace0=go.Box(y=votes_yes,name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))

votes_no=df[df['online_order']=="No"]['votes']
trace1=go.Box(y=votes_no,name="Not accepting online orders",
              marker = dict(
        color = 'rgb(0, 128, 128)',
    ))

layout = go.Layout(
    title = "Box Plots of votes",width=700,height=500
)

data=[trace0,trace1]
fig=go.Figure(data=data,layout=layout)

fig.write_html(city_name+"/html3.html")

image_name = 'Votes'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Box plots of votes for online vs offline orders',
                       'path': 'html3.html', 'type':'html'}


py.plot(fig, auto_open = False)


plt.figure(figsize=(10,10))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.xlabel("count")
plt.savefig(city_name+'/img5.png')


image_name = 'Restaurant Types'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Types of restaurants in the city',
                       'path': 'img5.png', 'type':'image'}


trace0=go.Box(y=df['approx_cost'],name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))
data=[trace0]
layout=go.Layout(title="Box plot of approximate cost",width=700,height=500,yaxis=dict(title="Price"))
fig=go.Figure(data=data,layout=layout)


fig.write_html(city_name+"/html4.html")

image_name = 'Approx Cost'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Box plot for approximate cost for two people',
                       'path': 'html4.html', 'type':'html'}

py.plot(fig, auto_open = False)


cost_dist=df[['rate','approx_cost','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

def return_budget(location,rest, cost_for_two):
    budget=cost_dist[(cost_dist['approx_cost']<=cost_for_two) & (cost_dist['location']==location) & 
                     (cost_dist['rate']>4) & (cost_dist['rest_type']==rest)]
    return(budget['name'].unique())


plt.figure(figsize=(10,10))
Rest_locations=df['location'].value_counts()[:20]
sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")
plt.savefig(city_name+'/img6.png')
plt.close()

image_name = 'Location Histogram'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Food Hotspots in the city',
                       'path': 'img6.png', 'type':'image'}




df_1=df.groupby(['location','cuisines']).agg('count')
data=df_1.sort_values(['url'],ascending=False).groupby(['location'],
                as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})


locations=pd.DataFrame({"Name":df['location'].unique()})
locations['Name']=locations['Name'].apply(lambda x: city_name + " " + str(x))
lat_lon=[]
geolocator=Nominatim(user_agent="app")
for ind in tqdm(range(locations['Name'].shape[0])):
    location = locations['Name'][ind]
    try:
      location = geolocator.geocode(location)
    except:
      location = None
    if location is None:
        lat_lon.append(np.nan)
    else:    
        geo=(location.latitude,location.longitude)
        lat_lon.append(geo)


locations['geo_loc']=lat_lon

locations["Name"]=locations['Name'].apply(lambda x :  x.replace(city_name,"")[1:])

Rest_locations=pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.columns=['Name','count']
Rest_locations=Rest_locations.merge(locations,on='Name',how="left").dropna()

zero_count = len(df[df.latitude == 0])
lat_mean = df['latitude'].sum()/(len(df) - zero_count) 
long_mean = df['longitude'].sum()/(len(df) - zero_count)

def generateBaseMap(default_location=[lat_mean, long_mean], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

lat,lon=zip(*np.array(Rest_locations['geo_loc']))
Rest_locations['lat']=lat
Rest_locations['lon']=lon
basemap=generateBaseMap()
HeatMap(Rest_locations[['lat','lon','count']].values.tolist(),max_zoom=20,radius=15).add_to(basemap)

basemap.save(city_name+'/html5.html')

image_name = 'Restaurant count'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Heatmap of restaurant count in the city',
                       'path': 'html5.html', 'type':'html'}


plt.figure(figsize=(10,10))
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.savefig(city_name+'/img7.png')
plt.close()
image_name = 'Most Popular Cuisines'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Most popular cuisines in the city',
                       'path': 'img7.png', 'type':'image'}

def produce_data(col,name):
    data= pd.DataFrame(df[df[col]==name].groupby(['location'],as_index=False)['url'].agg('count'))
    data.columns=['Name','count']
    data=data.merge(locations,on="Name",how='left').dropna()
    data['lan'],data['lon']=zip(*data['geo_loc'].values)
    return data.drop(['geo_loc'],axis=1)


North_India=produce_data('cuisines','North Indian')

basemap=generateBaseMap()
HeatMap(North_India[['lan','lon','count']].values.tolist(),max_zoom=20,radius=15).add_to(basemap)

basemap.save(city_name+'/html6.html')

image_name = 'North Indian Restaurants Count'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Heatmap of north indian restaurant count in the city',
                       'path': 'html6.html', 'type':'html'}

food=produce_data('cuisines','South Indian')
basemap=generateBaseMap()
HeatMap(food[['lan','lon','count']].values.tolist(),max_zoom=20,radius=15).add_to(basemap)

basemap.save(city_name+'/html7.html')

image_name = 'South Indian Restaurants Count'
fig_dat[image_name] = {'name': image_name, 'longtext': 'Heatmap of south indian restaurant count in the city',
                       'path': 'html7.html', 'type':'html'}


def produce_chains(name):
    data_chain=pd.DataFrame(df[df["name"]==name]['location'].value_counts().reset_index())
    data_chain.columns=['Name','count']
    data_chain=data_chain.merge(locations,on="Name",how="left").dropna()
    data_chain['lan'],data_chain['lon']=zip(*data_chain['geo_loc'].values)
    return data_chain[['Name','count','lan','lon']]


df_1=df.groupby(['rest_type','name']).agg('count')
datas=df_1.sort_values(['url'],ascending=False).groupby(['rest_type'],
                as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})

df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])

rest=df['rest_type'].value_counts()[:9].index
def produce_wordcloud(rest):
    
    plt.figure(figsize=(30,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=df[df['rest_type']==r]['dish_liked'].values.tolist()
        corpus=','.join(x  for list_words in corpus for x in list_words)
        
        try:
            wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                          width=1500, height=1500).generate(corpus)
        except:
            pass
        
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        
    image_name = 'Popular dishes'
    
    plt.savefig(city_name+'/img8.png')
    fig_dat[image_name] = {'name': image_name, 'longtext': 'Wordcloud of popular dishes',
                           'path': 'img8.png', 'type':'image'}
      
produce_wordcloud(rest)

all_ratings = []

for name,ratings in tqdm(zip(df['name'],df['reviews'])):
    ratings = eval(ratings)
    for score, doc in ratings:
        if score:
            score = score.strip("Rated").strip()
            doc = doc.strip('RATED').strip()
            score = float(score)
            all_ratings.append([name,score, doc])

rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])
rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))

rest=df['name'].value_counts()[:9].index
#def produce_wordcloud(rest):
    
    #plt.figure(figsize=(20,20))
    #for i,r in enumerate(rest):
        #plt.subplot(3,3,i+1)
        #corpus=rating_df[rating_df['name']==r]['review'].values.tolist()
        #corpus=' '.join(x  for x in corpus)
        #wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      #width=1500, height=1500).generate(corpus)
        #plt.imshow(wordcloud)
        #plt.title(r)
        #plt.axis("off")
        

    #image_name = 'Reviews'
    
    #plt.savefig(city_name+'/img9.png')
    #fig_dat[image_name] = {'name': image_name, 'longtext': 'Wordcloud of reviews of various restaurants',
                           #'path': 'img9.png', 'type':'image'} 
        
#produce_wordcloud(rest)


plt.figure(figsize=(10,10))
rating=rating_df['rating'].value_counts()
sns.barplot(x=rating.index,y=rating)
plt.xlabel("Ratings")
plt.ylabel('Count')


image_name = 'Ratings'
    
plt.savefig(city_name+'/img10.png')
fig_dat[image_name] = {'name': image_name, 'longtext': 'Ratings given by user',
                       'path': 'img10.png', 'type':'image'}


#fig_dat['ratings_hist'] = {}
#fig_dat['ratings_hist']['title'] = "Histogram of Ratings"
#fig_dat['ratings_hist']['long_text'] = "Histogram displaying most famous restaurants chains in " + city_name

data = {}
data['count'] = len(df)
data['data'] = fig_dat

with open(city_name+'/data.json', 'w') as fp:
    json.dump(data, fp)


print('\n\n******DataAnalyzer has analzyed all the data. Now launcing data uplaoder*****')

from Naked.toolshed.shell import execute_js, muterun_js

result = execute_js('DataUploader.js' + ' ' + city_name)
