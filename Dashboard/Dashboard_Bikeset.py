import os
import sys

try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except ModuleNotFoundError as e:
    print(f"Error: {e}. Please ensure all required packages are installed.", file=sys.stderr)
    sys.exit(1)

# Set page config
st.set_page_config(
    page_title="Bike Sharing Analysis",
    page_icon="🚲",
    layout="wide"
)

# Set relative path for data loading
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, '..', 'Bike Sharing Dataset')

# Load and clean datasets
try:
    data_hour = pd.read_csv(os.path.join(data_dir, 'hour.csv'))
    data_day = pd.read_csv(os.path.join(data_dir, 'day.csv'))
except FileNotFoundError as e:
    st.error(f"Error loading data: {e}. Please check if the data files exist in the correct location.")
    st.stop()

# Function to remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Clean datasets
numerical_columns = data_day.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_hour = data_hour.select_dtypes(include=['float64', 'int64']).columns

cleaned_data_day = data_day.copy()
cleaned_data_hour = data_hour.copy()

for column in numerical_columns:
    cleaned_data_day = remove_outliers_iqr(cleaned_data_day, column)
for column in numerical_columns_hour:
    cleaned_data_hour = remove_outliers_iqr(cleaned_data_hour, column)

# Sidebar filters
st.sidebar.title("Bike Sharing Analysis Dashboard")

# Date filter
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(cleaned_data_day['dteday']).min())
end_date = st.sidebar.date_input("End Date", pd.to_datetime(cleaned_data_day['dteday']).max())

# Season filter
season_dict = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
selected_season = st.sidebar.multiselect(
    "Select Season",
    options=list(season_dict.keys()),
    default=list(season_dict.keys()),
    format_func=lambda x: season_dict[x]
)

# Filter data based on selections
filtered_data_day = cleaned_data_day[
    (pd.to_datetime(cleaned_data_day['dteday']) >= pd.to_datetime(start_date)) & 
    (pd.to_datetime(cleaned_data_day['dteday']) <= pd.to_datetime(end_date)) &
    (cleaned_data_day['season'].isin(selected_season))
]

filtered_data_hour = cleaned_data_hour[
    (pd.to_datetime(cleaned_data_hour['dteday']) >= pd.to_datetime(start_date)) & 
    (pd.to_datetime(cleaned_data_hour['dteday']) <= pd.to_datetime(end_date)) &
    (cleaned_data_hour['season'].isin(selected_season))
]

# Main content
st.title("Bike Sharing Analysis Dashboard")

# Question 1: Environmental and Seasonal Analysis
st.header("1. Environmental and Seasonal Factors Impact")

# Seasonal distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rental Distribution by Season")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_data_day, x='season', y='cnt', ax=ax1)
    ax1.set_title("Jumlah Peminjaman Berdasarkan Musim")
    ax1.set_xlabel("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)")
    ax1.set_ylabel("Jumlah Peminjaman")
    st.pyplot(fig1)

with col2:
    st.subheader("Environmental Factors Correlation")
    environmental_columns = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_data_day[environmental_columns].corr(), 
                annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Question 2: Usage Patterns Analysis
st.header("2. Usage Patterns Analysis")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Average Usage by Day Type")
    usage_by_workingday = filtered_data_day.groupby('workingday')[['casual', 'registered', 'cnt']].mean()
    st.bar_chart(usage_by_workingday)

with col4:
    st.subheader("Hourly Usage Patterns")
    hourly_usage = filtered_data_hour.groupby('hr')[['casual', 'registered']].mean()
    st.line_chart(hourly_usage)

# Additional Statistics
st.header("Summary Statistics")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Total Rentals")
    total_rentals = filtered_data_day['cnt'].sum()
    st.metric("Total Bike Rentals", f"{total_rentals:,}")
    
with col6:
    st.subheader("Average Daily Rentals")
    avg_rentals = filtered_data_day['cnt'].mean()
    st.metric("Average Daily Rentals", f"{avg_rentals:,.0f}")

# Add footer with data source information
st.markdown("---")
st.markdown("Data source: Bike Sharing Dataset")
st.markdown("Analysis by: Steven Hot Asi Sihite")