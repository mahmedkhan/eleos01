# Importing the pandas and numpy libraries, as well as particular modules from sklearn.
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing our resource database.
df = pd.read_csv('resources.csv')
df.head()

while True:
    action = input("What do you want me to do? ")
    if action == "Find me the best resources for managing grief.":
        # Creating a new dataframe that contains resources with the keyword `grief`
        dfS = df[df['tags'].str.contains("grief")]

        # Calculating the average vote average for each resource in the dataframe
        C = dfS['vote_average'].mean()

        # Calculating the 30th percentile for the vote count of all resources in the dataframe
        m = dfS['vote_count'].quantile(0.3)

        # Creating a new dataframe containing only resources that meet the vote count threshold
        dfQ = dfS.copy().loc[dfS['vote_count'] >= m]

        # Weighting the vote count and average for each resource
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            # Calculation based on the IMDB formula
            return (v / (v + m) * R) + (m / (m + v) * C)

        # Defining a 'score' and calculating its value with `weighted_rating()`
        dfQ['score'] = dfQ.apply(weighted_rating, axis=1)

        # Sorting the qualifying recommendations based on their score
        dfQ = dfQ.sort_values('score', ascending=False)

        # Printing the top recommendations
        print('')
        print(dfQ[['name', 'url']].head())
        print('')

    elif action == "I have been feeling pretty stressed at school lately. Can you help me?":
        # Importing the MonkeyLearn Keyword Extraction algorithm
        from monkeylearn import MonkeyLearn

        # Authenticating with MonkeyLearn
        ml = MonkeyLearn('da0088bcfbb0add6f5cccff301a6d98bfbac4d77')
        data = ["School has been stressing me out lately."]
        model_id = 'ex_YCya9nrn'
        result = ml.extractors.extract(model_id, data)

        from pandas.io.json import json_normalize

        dfB = pd.json_normalize(result.body)

        dfB = pd.json_normalize(dfB.iloc[0]['extractions']).head()
        dfB.rename(columns={'parsed_value': 'keywords'}, inplace=True)
        dfB = dfB[['keywords', 'relevance']]

        # Python3 code to remove whitespace
        def strip(keyword):
            return keyword.replace(" ", "")

        keyword = strip(dfB.iloc[0]['keywords']).lower()
        dfL = df[df['tags'].str.contains(keyword)]

        print("")
        print("Okay, here is a list of resources for managing " + "" + keyword + ".")
        print("")
        print(dfL[['name', 'url']].head())
        print('')

    elif action == "Find me resources like Wysa.":
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['tags'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        indices = pd.Series(df['name'])

        def recommend(name, cosine_sim=cosine_sim):
            recommendations = []
            idx = indices[indices == name].index[0]
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
            top = list(score_series.iloc[1:4].index)

            for i in top:
                recommendations.append(list(df['name'])[i])

            dfR = df[df['name'].isin([recommendations[0], recommendations[1], recommendations[2]])]
            dfR = dfR[['name', 'url']]
            return dfR

        dfR = recommend('Wysa')
        print("")
        print(dfR)
        print('')
    elif action == "Find me the best resources for grief that people similar to me have found useful.":
        from surprise import Reader
        from surprise import SVD
        from surprise import Dataset
        from surprise.model_selection import cross_validate

        reader = Reader()
        ratings = pd.read_csv('ratings.csv')
        ratings.head()

        data = Dataset.load_from_df(ratings[['u_id', 'r_id', 'rating']], reader)
        algo = SVD()
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

        trainset = data.build_full_trainset()
        algo.fit(trainset)

        u_id = 3 #int(input('Ask me to predict the ratings for a user: '))
        # ratings[ratings['u_id'] == u_id]

        predictions = []
        for i in range(0, len(df)):
            prediction = algo.predict(u_id, i)
            predictions.append(prediction.est)

        dfP = pd.DataFrame(predictions, columns=["prediction"])
        dfP = pd.merge(df, dfP, left_index=True, right_index=True)
        dfP = dfP[dfP['tags'].str.contains("grief")]

        # Calculating the average vote average for each resource in the dataframe
        C = dfP['vote_average'].mean()

        # Calculating the 60th percentile for the vote count of all resources in the dataframe
        m = dfP['vote_count'].quantile(0.6)

        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            # Calculation based on the IMDB formula
            return (v / (v + m) * R) + (m / (m + v) * C)

        # Defining a 'score' and calculating its value with `weighted_rating()`
        dfP['score'] = dfP.apply(weighted_rating, axis=1)

        # Weighting the vote count and average for each qualifying resource
        def weighted_prediction(x):
            score = x['score']
            prediction = x['prediction']
            return score * prediction

        # Defining a 'score' and calculating its value with `weighted_rating()`
        dfP['prediction'] = dfP.apply(weighted_prediction, axis=1)

        # Sorting the qualifying recommendations based on their score
        dfP = dfP.sort_values('prediction', ascending=False)

        # Printing the top recommendations
        print("")
        print(dfP[['name', 'url']].head())
        print('')
    elif action == "That is all, thank you.":
        print("No worries, glad to help!")
        print("- Eleos")
        break
    else:
        print("I didn't catch that, maybe try again?")
