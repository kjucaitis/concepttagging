import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import seaborn as sns
#from mpl_toolkits.basemap import TooltipBasemap

%matplotlib inline

slice_wrong = pd.read_csv("./Data/slice_level_oh_tags.csv", low_memory=False)
meta_data = pd.read_csv("./Data/concept_tagging_metadata.csv", low_memory=False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 35)

slice = slice_wrong.copy()

slice['distinctiveness_mean'] = 6 - slice['distinctiveness_mean']

slice

meta_data

slice.columns

slice['advantage_mean'].rank()

metrics = ['advantage_mean', 'distinctiveness_mean', 'believability_mean', 'purchase_likelihood_mean', 'premiumness_mean']

slice[metrics].hist(figsize=(16,8))

'''
import pandas as pd
from itertools import combinations

def find_combinations(dataframe, exclude_columns):
    column_names = [col for col in dataframe.columns if col not in exclude_columns]
    combinations_list = []

    for r in range(1, len(column_names) + 1):
        for combo in combinations(column_names, r):
            combinations_list.append(combo)

    return combinations_list

combinations_list = find_combinations(slice, metrics)

result_rows = []

for combo in combinations_list:
    combo_slice = slice[list(combo)]
    rows_with_all_ones = combo_slice[(combo_slice == 1).all(axis=1)]

    if not rows_with_all_ones.empty:
        print(f"Combo {combo}:")
        print(rows_with_all_ones)  # Print matching rows for this combination
        result_rows.append(rows_with_all_ones)

result_df = pd.concat(result_rows, ignore_index=True)

print("Result DataFrame:")
print(result_df)
'''

split_cols = [col.split('_mapped_clean_') for col in slice.columns]
split_cols

# Define the keywords you want to search for
keywords = metrics + ['concept_id']
print(keywords)
# Filter columns that contain any of the keywords
#tag_filter = [col for col in slice.columns if any(keyword in col for keyword in keywords)]

# Create a new dataframe with only the filtered columns
#tags_slice = slice[tag_filter]

tags_slice = slice.copy()
# Create an empty list to store flavors
tags = []

# For every column which includes a flavor replace name with that flavor and save to flavors list
for str in tags_slice.columns:
    if '_mapped_clean_' in str:
        #tag = str.replace('_mapped_clean_', '')
        tags.append(str)

# Rename columns dynamically
#tags_slice.columns = tags_slice.columns.str.replace('names_mapped_clean_', '')

tags_slice

# Creates an empty list to store where two tags exist at the same time
tag_combinations = {}
combination_count = 0

for i in range(len(tags_slice)):  # Iterate through rows
    for tag in tags:
        if tags_slice[tag].iloc[i] == 1:  # Check if the current row has tag = 1
            for tag2 in tags:
                if tag != tag2 and tags_slice[tag2].iloc[i] == 1:  # Check if the current row has tag2 = 1
                    combination_count += 1
                    tag_combinations[tag] = tag2

print("Number of combinations:", combination_count)

print(tag_combinations)

for key, value in tag_combinations.items():
    print(f"{key}: {value}")


import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Iterate through the tag combinations and add them to the graph as edges
for key, value in tag_combinations.items():
    G.add_edge(key, value)

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)

# Adjust font size and label positioning
labels = {}


for node in G.nodes():
    labels[node] = node  # This assigns the node's name as the label

nx.draw(G, pos, labels=labels, node_size=50, font_size=6, node_color='skyblue', edge_color='gray', font_color='black', alpha=0.7)
plt.title("Tag Combinations")
plt.show()


'''
edge_trace = []
node_trace = []

for key, value in tag_combinations.items():
    edge_trace.append[key]
    node_trace.append[value]

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
'''

import pandas as pd

# Initialize a list to store metrics for each combination
combination_metrics = []

desired_order = ['advantage_mean', 'distinctiveness_mean', 'believability_mean', 'purchase_likelihood_mean', 'premiumness_mean']

def calculate_combined_metric(filtered_df, metrics):
    result = {}
    
    for metric in metrics:
        metric_values = filtered_df[metric].values
        result[metric] = metric_values.mean()  # Calculate the metric as needed
        
    return result

# Loop through the tag combinations
for tag1, tag2 in tag_combinations.items():
    # Filter rows where both tags are equal to 1
    filtered_df = tags_slice[(tags_slice[tag1] == 1) & (tags_slice[tag2] == 1)]
    
    result = calculate_combined_metric(filtered_df, desired_order)
    #print(result)
    
    # Append the metric values to the combination_metrics list
    # Include tag1 and tag2 only in the row where they are both 1
    if not filtered_df.empty:
        combination_metrics.append({'Tag1': tag1, 'Tag2': tag2, **result})


print(combination_metrics)

# Create a DataFrame from the combination metrics
combination_metrics_df = pd.DataFrame(combination_metrics)

# Display the resulting DataFrame
combination_metrics_df

from scipy import stats

# Creates dictionary to score P-values

P_values_tags = {}

for tag in tags:
    for metric in metrics:
        # Separate distinctiveness_mean for Beef and Jalapeno
        metric_tag = tags_slice[tags_slice[tag] == 1][metric]
        metric_not_tag = tags_slice[tags_slice[tag] == 0][metric]
        
        # Perform a t-test to compare premiumness between Beef and Jalapeno
        t_stat, p_value = stats.ttest_ind(metric_tag, metric_not_tag)

        
        # Output the results
        print(tag)
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
        P_values_tags[f"{tag} {metric}"] = p_value
        
        # Check if the difference is statistically significant at a significance level of 0.05
        if p_value < 0.05:
            print(f"There is a statistically significant difference between {tag} and Not {tag} {metric}.")
        else:
            print(f"There is no statistically significant difference between {tag} and Not {tag} {metric}.")



print("\n", P_values_tags.items())
print("\nTag and Metric with most significant difference: ", min(P_values_tags, key=P_values_tags.get))


# Find the 10 minimum values and their corresponding keys
ten_min_items = sorted(P_values_tags.items(), key=lambda x: x[1])[:10]

print("Ten most significant Tag and Metric pairs:")
for key, value in ten_min_items:
    print(f"{key}: {value}")

marked = []

print("Ten most significant unique Tag and Metric pairs:")
for key, value in ten_min_items:
    if key not in marked:
        print(f"{key}: {value}")
        marked.append(key)

import matplotlib.pyplot as plt
import numpy as np  # Import numpy for logarithmic scaling

# Assuming you have defined min_items, metrics, and tags lists
# (min_items is a list of (key, value) pairs, metrics and tags are lists of strings)

# Find the 10 minimum values and their corresponding keys
min_items = sorted(P_values_tags.items(), key=lambda x: x[1])

# Iterate through the metrics
for metric in metrics:
    print("\nTen most significant Tags for " + metric + ":\n")
    count = 0
    metric_tags = []  # Store the tags for the current metric
    metric_values = []  # Store the corresponding values for the current metric

    # Iterate through the min_items, stopping after 10 iterations for each metric
    for key, value in min_items:
        if metric in key:
            print(f"{key}: {1 - value}")  # Reverse the significance value
            count += 1
            metric_tags.append(key)
            metric_values.append(1 - value)  # Reverse the significance value
            if count >= 10:
                break

    # Create a bar chart for the current metric using Seaborn and Pandas
    plt.figure(figsize=(10, 6))
    plt.title(f"Ten most significant Tags for {metric}")
    plt.barh(metric_tags, metric_values, color='skyblue')
    plt.xlabel("Reversed P-value")
    plt.ylabel("Tag")
    plt.xscale('log')  # Use logarithmic scaling for the x-axis
    plt.show()


advantage_multipack = tags_slice[tags_slice['multipack or single serving_mapped_clean_Multipack'] == 1]['advantage_mean'].mean()
advantage_no_multipack = tags_slice[tags_slice['multipack or single serving_mapped_clean_Multipack'] == 0]['advantage_mean'].mean()

print("Advantage mean with Multipack: ", advantage_multipack)
print("Advantage mean without Multipack: ", advantage_no_multipack)

print(advantage_multipack/advantage_no_multipack)
print(advantage_multipack - advantage_no_multipack)

tags_slice['multipack or single serving_mapped_clean_Multipack'].sum()

import matplotlib.pyplot as plt

tags_slice.boxplot(column='advantage_mean', by='multipack or single serving_mapped_clean_Multipack', vert=False)
plt.title('Box Plot of Advantage Mean for Multipack Products')
plt.xlabel('multipack or single serving_mapped_clean_Multipack')
plt.ylabel('Advantage Mean')
plt.suptitle('')  # Remove the default title created by Pandas
plt.show()

#advantage_multipack.hist(figsize=(16,8))

#advantage_no_multipack.hist(figsize=(16,8))

import seaborn as sns

sns.barplot(data=tags_slice, x='ingredients mentioned in the flavor names_mapped_clean_Cheese', y="advantage_mean", hue="multipack or single serving_mapped_clean_Multipack")

sns.pairplot(tags_slice[metrics + ['multipack or single serving_mapped_clean_Multipack']], hue='multipack or single serving_mapped_clean_Multipack')

sns.pairplot(tags_slice[metrics + ['multipack or single serving_mapped_clean_Multipack']], hue='multipack or single serving_mapped_clean_Multipack', kind='kde')

flavors_slice.sum(axis=0)

flavors_slice.Cheese.sum()

flavors_slice.drop(columns=['concept_id']).sum(axis=0)

# Creates dictionary to score P-values

P_values_flavors = {}

for flavor in flavors:
    for metric in metrics:
        # Separate distinctiveness_mean for Beef and Jalapeno
        metric_flavor = flavors_slice[flavors_slice[flavor] == 1][metric]
        metric_not_flavor = flavors_slice[flavors_slice[flavor] == 0][metric]
        
        # Perform a t-test to compare premiumness between Beef and Jalapeno
        t_stat, p_value = stats.ttest_ind(metric_flavor, metric_not_flavor)

        
        # Output the results
        print(flavor)
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
        P_values_flavors[f"{flavor} {metric}"] = p_value
        
        # Check if the difference is statistically significant at a significance level of 0.05
        if p_value < 0.05:
            print(f"There is a statistically significant difference between {flavor} and Not {flavor} {metric}.")
        else:
            print(f"There is no statistically significant difference between {flavor} and Not {flavor} {metric}.")



print("\n", P_values_flavors.items())
print("\nFlavor and Metric with most significant difference: ", min(P_values_flavors, key=P_values_flavors.get))


# Find the 10 minimum values and their corresponding keys
min_items = sorted(P_values_flavors.items(), key=lambda x: x[1])[:10]

print("Ten most significant Flavor and Metric pairs:")
for key, value in min_items:
    print(f"{key}: {value}")


matthews_corrcoef(flavors_slice['Beef'], flavors_slice['Sweet Chili'])

purchase_likelihood_cheese = flavors_slice[flavors_slice['Cheese'] == 1]['purchase_likelihood_mean'].mean()
purchase_likelihood_no_cheese = flavors_slice[flavors_slice['Cheese'] == 0]['purchase_likelihood_mean'].mean()

print("Purchase likelihood mean with Cheese: ", purchase_likelihood_cheese)
print("Purchase likelihood mean without Cheese: ", purchase_likelihood_no_cheese)

print(purchase_likelihood_cheese/purchase_likelihood_no_cheese)
print(purchase_likelihood_cheese - purchase_likelihood_no_cheese) 

# Create separate box plots for Herbs and Spices = 0 and Herbs and Spices = 1
flavors_slice.boxplot(column='premiumness_mean', by='Nuts', vert=False)
plt.title('Box Plot of Premiumness Mean for Nuts')
plt.xlabel('Nuts')
plt.ylabel('Premiumness Mean')
plt.suptitle('')  # Remove the default title created by Pandas
plt.show()

flavors_slice.to_csv('Data/flavors_slice.csv')
tags_slice.to_csv('Data/tags_slice.csv')

sns.pairplot(flavors_slice[['advantage_mean', 'distinctiveness_mean', 'Cheese']], hue='Cheese')

combination_metrics_df

print(combination_metrics)

correlation_matrix = combination_metrics_df[metrics].corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add labels and title
plt.title("Correlation Heatmap")
plt.xlabel("Metrics")
plt.ylabel("Metrics")

# Show the heatmap
plt.show()

shit_to_drop = [ 'concept_id',
    'advantage_mean', 'advantage_stdev', 'advantage_median',
    'believability_mean', 'believability_stdev', 'believability_median',
    'distinctiveness_mean', 'distinctiveness_stdev', 'distinctiveness_median',
    'purchase_likelihood_mean', 'purchase_likelihood_stdev', 'purchase_likelihood_median',
    'premiumness_mean', 'premiumness_stdev', 'premiumness_median',
    'advantage_top_box', 'distinctiveness_top_box',
    'purchase_likelihood_top_box',
    'believability_top_box',
    'premiumness_top_box'
] + metrics

tags_slice = slice.copy()

print(metrics)

ts = tags_slice.drop(shit_to_drop, axis=1)

ts

# Calculate the pairwise correlation matrix
ts_correlation_matrix = ts.corr()

# Create a heatmap of the binary DataFrame
plt.figure(figsize=(50,10))  # Set the figure size as desired
sns.heatmap(ts, cmap='coolwarm', linecolor='blue', cbar=True, annot=False, vmax=1, vmin=0)  # Use a colormap of your choice
plt.title("Binary Data Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()

# Calculate the pairwise correlation matrix
ts_correlation_matrix = ts.corr()

# Filter the correlation matrix to keep only the top 10% correlations
top_10_percent = ts_correlation_matrix.stack().quantile(0.9)
ts_correlation_matrix_filtered = ts_correlation_matrix.mask(
    np.abs(ts_correlation_matrix) < top_10_percent, np.nan
)

# Create a heatmap of the filtered correlation matrix
plt.figure(figsize=(10, 10))  # Set the figure size as desired
sns.heatmap(
    ts_correlation_matrix_filtered,
    cmap='rainbow',
    linecolor='blue',
    cbar=True,
    annot=False,
    vmax=1,
    vmin=0,
)  # Use a colormap of your choice
plt.title("Binary Data Heatmap (Top 10% Correlations)")
plt.xlabel("Columns")
plt.ylabel("Rows")

plt.show()


'''
from scipy import stats

# Creates dictionary to score P-values

P_values_tags_composite = {}

for i, (tag1, tag2, correlation) in enumerate(sorted_positive_pairs):
    for metric in metrics:
        # Separate distinctiveness_mean for Beef and Jalapeno
        metric_tag = tags_slice[tags_slice[tag] == 1][metric]
        metric_not_tag = tags_slice[tags_slice[tag] == 0][metric]
        
        # Perform a t-test to compare premiumness between Beef and Jalapeno
        t_stat, p_value = stats.ttest_ind(metric_tag, metric_not_tag)

        
        # Output the results
        print(tag)
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
        P_values_tags_composite[f"{tag} {metric}"] = p_value
        
        # Check if the difference is statistically significant at a significance level of 0.05
        if p_value < 0.05:
            print(f"There is a statistically significant difference between {tag} and Not {tag} {metric}.")
        else:
            print(f"There is no statistically significant difference between {tag} and Not {tag} {metric}.")



print("\n", P_values_tags_composite.items())
print("\nTag and Metric with most significant difference: ", min(P_values_tags_composite, key=P_values_tags_composite.get))


# Find the 10 minimum values and their corresponding keys
min_items = sorted(P_values_tags_composite.items(), key=lambda x: x[1])[:10]

print("Ten most significant Tag and Metric pairs:")
for key, value in min_items:
    print(f"{key}: {value}")
'''

combination_metrics_df



