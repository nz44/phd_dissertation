# GWU Economcis Ph.D. Dissertation 
My dissertation analyze whether a product is niche will affect its freemium pricing strategies. 
The main set up is difference-in-difference with Covid-19 stay-at-home order as the cut-off event, the treatment groups being niche apps and the control group being the broad apps. 
The niche apps are determined by k-means clustering of word matrices of apps' descriptions. 
The pricing strategies are a few dummy variables including whether the app has in-app purchases, whether the app is advertisement supported and a continous price variable. 
Essay 1 uses the entire sample, Essay 2 uses the sub-sample of market leading apps, and Essay 3 uses the sub-sample of market follower apps. 


## script folder
In order to run the project, start with ___essay_1_main_code___.ipynb, it has sections and calls classes and methods from STEP1 to STEP5 respectively. For each class object, one need to change the class attribute path to your local path. 

For Essay 2 and Essay 3, ___essay_2_main_code___.ipynb and ___essay_3_main_code___.ipynb calls their respective classes and methods from STEP6 to STEP8 respectively. 


___essay_1_test_code___.ipynb, ___essay_2_test_code___.ipynb, ___essay_3_test_code___.ipynb are playground scripts for testing code snippets. 

## output folders

overall_graphs folder contains the graphs output of parallel trends that put essay 2 and essay 3 samples into the same graph. All the other folders containing descriptive statistics, natural language processing (nlp), regression results, results tables in LaTex and graphs outputs relating to each essay's sub-samples. 

