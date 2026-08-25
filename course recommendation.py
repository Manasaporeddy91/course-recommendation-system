# Course Recommendation System using ML

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Dataset (Courses + Skills)
# -------------------------------
data = {
    'course': [
        'Machine Learning',
        'Data Science',
        'Web Development',
        'Deep Learning',
        'Android Development',
        'Cyber Security',
        'Cloud Computing'
    ],
    'skills': [
        'python statistics machine learning',
        'python data analysis visualization',
        'html css javascript web',
        'neural networks deep learning python',
        'java kotlin android',
        'network security ethical hacking',
        'aws cloud devops'
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Convert text to vectors
# -------------------------------
cv = CountVectorizer()
vectors = cv.fit_transform(df['skills'])

# -------------------------------
# Step 3: Similarity matrix
# -------------------------------
similarity = cosine_similarity(vectors)

# -------------------------------
# Step 4: Recommendation function
# -------------------------------
def recommend(course_name):
    if course_name not in df['course'].values:
        print("Course not found!")
        return
    
    index = df[df['course'] == course_name].index[0]
    distances = list(enumerate(similarity[index]))
    
    # Sort based on similarity score
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    
    print("\nRecommended Courses:")
    for i in distances[1:4]:
        print(df.iloc[i[0]].course)

# -------------------------------
# Step 5: User Input
# -------------------------------
print("Available Courses:")
print(df['course'].to_string(index=False))

user_input = input("\nEnter a course you like: ")

recommend(user_input)
