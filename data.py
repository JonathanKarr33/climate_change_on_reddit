import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read the data
df = pd.read_csv("data/the-reddit-climate-change-dataset-comments.csv")

# Sample 1/1000 of the data with a seed of 42
sampled_df = df.sample(frac=0.001, random_state=42)

# Convert 'created_utc' to datetime
sampled_df['created_utc'] = pd.to_datetime(sampled_df['created_utc'], unit='s')

# Extract month from 'created_utc'
sampled_df['month'] = sampled_df['created_utc'].dt.to_period('M')

# Group by month and count the number of posts
posts_per_month = sampled_df.groupby('month').size()

# Plot the number of posts during each month with color-coded sentiment scores
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(posts_per_month.index)),
               posts_per_month, color='gray')
plt.xlabel('Month')
plt.ylabel('Number of Posts')
plt.title('Number of Posts During Each Month')

# Set x-axis labels
# Set x-axis labels and tick locations
x_ticks_locations = range(0, len(posts_per_month.index), 3)
x_ticks_labels = posts_per_month.index.strftime('%Y-%m')[::3]
plt.xticks(x_ticks_locations, x_ticks_labels, rotation=45)

# Define color map for sentiment scores
color_map = {
    -1.0: 'red',
    -0.75: 'orange',
    -0.5: 'yellow',
    -0.25: 'lightgreen',
    0.0: 'green',
    0.25: 'lightblue',
    0.5: 'blue',
    0.75: 'darkblue',
    1.0: 'purple'
}

# Color code bars based on sentiment scores
for i, bar in enumerate(bars):
    month = posts_per_month.index[i]
    mean_sentiment = sampled_df[sampled_df['month']
                                == month]['sentiment'].mean()
    for sentiment, color in color_map.items():
        if mean_sentiment <= sentiment:
            bar.set_color(color)
            break

plt.tight_layout()
plt.show()

# Encode month as categorical variables
sampled_df = pd.get_dummies(sampled_df, columns=['month'])

# Define independent variables (features)
X = sampled_df.drop(columns=['sentiment', 'score'])

# Add constant to independent variables (for intercept)
X = sm.add_constant(X)

# Define dependent variable (target)
y = sampled_df['sentiment']

# Perform linear regression
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())
