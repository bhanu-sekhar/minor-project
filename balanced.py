import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
csv_file_path = 'captions_small.csv'
df = pd.read_csv(csv_file_path)

# df = pd.read_csv('captions_small.csv', engine='python')


unique_genres = df['genre'].unique()
print(unique_genres)
genre_mapping = {
    'fun':'fun',
    'fun ':'fun',
    'fun  ':'fun',
    'happy':'happy',
    'happy ':'happy',
    'excited':'excited',
    'excited ':'excited',
    'excited  ':'excited',
    'blog':'blog',
    'blog ':'blog',
    'blog  ':'blog'
}

df['genre'] = df['genre'].map(genre_mapping).fillna(df['genre'])

unique_genres = df['genre'].unique()
print(unique_genres)



# Group by genre and count the number of captions
genre_counts = df.groupby('genre').size().reset_index(name='caption_count')

# Display the distribution
print(genre_counts)

# Bar plot for caption counts per genre
sns.barplot(data=genre_counts, x='genre', y='caption_count')
plt.title('Caption Count per Genre')
plt.show()

# Calculate statistics
mean_captions = genre_counts['caption_count'].mean()
median_captions = genre_counts['caption_count'].median()
std_dev_captions = genre_counts['caption_count'].std()

print(f"Mean Captions per Genre: {mean_captions}")
print(f"Median Captions per Genre: {median_captions}")
print(f"Standard Deviation of Captions: {std_dev_captions}")


