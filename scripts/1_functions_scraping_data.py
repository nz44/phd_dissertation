from google_play_scraper import app
from tqdm import tqdm
import numpy as np

# get ID list
# data scraped before 2020 Sep are organized in a list, each dictionary inside the list contains attributes and their values
# for the data scraped in 2020 Sep and onwards, they are organized in dictionary with key as appid, and then their appdetails,
# so id_list should just be C.keys()
def get_id_from_data_beofre_202009(C):
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




