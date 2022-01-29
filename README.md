# NFT-Auctions
Data collected from Kaggle: https://www.kaggle.com/c/stat440-21-project2
We are tackling is to accurately predict auction closing prices in Ethereum for a set of 13828 NFTs. 


We tried as many different types of models as we can to identify which has the most potential. The methods we used include Light Gradient Boosted Machine (LightGBM), Random Forest, Epsilon-Support Vector Regression (SVR), and Stochastic Gradient Descent (SGD). The Kaggle MAE on the public leaderboard for those 4 models are 15.12958, 9.52078, 9.33971, and 8. 93930 respectively. Thus, we can conclude that SGD seemed to perform best on the dataset


We created two different corpus, one is for ‘description’ and one is for ‘symbol’. Then we transform the text in corpus into a vector on the basis of the frequency(count) of each word that occurs in the entire. After that, we try to assign these texts into K topics and report the percentage of each topic on a dataframe. The topic modeling technique we used here is Latent Dirichlet Allocation(LDA), which is a typical model of Bag Of Word(BOG).


The given image file contains images of NFTs. To process them, we have to extract the height, width, the mean RGB value, the minimum RGB value, and the red, green, and blue layers. 


SGD is one of the supervised machine learning techniques that optimizes the cost function in the learning process. It is a simple and efficient approach to fitting linear classifieds and regressors under convex loss functions. The function we used in python is the SGDRegressor from package sklearn.linear_model. And the loss function used is epsilon_insensitive. It ignores errors less than epsilon and is linear past that. We tried SGD several times, including: with only 4 numeric variables (main); with main and text; with main and images; with main, text and images; with main, text, images, and version.
