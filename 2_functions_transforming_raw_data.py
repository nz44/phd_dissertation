# The file include functions that transform raw scraped app data into
# 1. categorical level and developer level

def transform_raw_to_category_data(raw_data):
    # get a list of unique categories in the raw_data
    categories = []
    for appid, appdetails in raw_data.items():
        if appdetails is not None:
            categories.append(appdetails['genreId'])
    categories = list(set(categories))
    appdetails_categories = dict.fromkeys(categories)

    for category in appdetails_categories.keys():
        content = []
        for appid, appdetails in raw_data.items():
            if appdetails is not None:
                if appdetails['genreId'] == category:
                    content.append(appdetails)
        appdetails_categories[category] = content

    for k, v in appdetails_categories.items():
        print(k, len(v))

    return(appdetails_categories)


# explore the summary stats within each category