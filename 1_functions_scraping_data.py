from google_play_scraper import app
from tqdm import tqdm
import numpy as np

# get ID lost
def get_id_from_old_data(C):
    id_list = []
    for i in C:
        id_list.append(i['appId'])
    return(id_list)

# scraping using google_play_scraper app function
# the input is from get_id_from_old_data
def scraping_apps_according_to_id(id_list):
    app_details = dict.fromkeys(id_list)
    # print(app_details)
    for j in tqdm(range(len(id_list)), desc="scraping..."):
        try:
            app_details[id_list[j]] = app(id_list[j])
        except:
            pass
    return(app_details)

# get a snapshot of the data you've scraped and extract one appdetail for closer examination
# the input appdetails are from scraping_apps_according_to_id
def get_a_glimpse(app_details):

    print('the dataset contains', len(app_details), 'unique apps.')

    random_number = np.random.randint(low=0, high=len(app_details), size=1)[0]
    list = []
    for key in app_details.keys():
        list.append(key)
    appid = list[random_number]

    print(appid)
    for k, v in app_details[appid].items():
        print(k, " : ", v)




