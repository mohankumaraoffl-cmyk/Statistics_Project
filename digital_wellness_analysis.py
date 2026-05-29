import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# 1. READ EXCEL DATASET FILE
file_name = 'Digital_Wellness_Dataset.xlsx'

if not os.path.exists(file_name):
    print(f"❌ Error: The file '{file_name}' was not found in the current directory!")
    print("Please ensure the Excel file is saved in the exact same folder as this Python script.")
else:
    # Reading Excel database using pandas engine
    df = pd.read_excel(file_name)
    print("✅ EXCEL DATA LOADED SUCCESSFULLY!")
    print(f"Total Number of Respondents (Sample Size N): {len(df)}")
    print("\n--- FIRST 5 ROWS OF THE DATASET ---")
    print(df.head())
    print("="*65)

    # ==========================================================
    # 2. DESCRIPTIVE STATISTICS (Data Summary Profiles)
    # ==========================================================
    print("\n--- DESCRIPTIVE STATISTICS SUMMARY ---")
    # Generating summary framework for central tendency metrics
    print(df[['Age', 'Screen_Time_Hours', 'Sleep_Duration_Hours']].describe().T)
    
    # Computing modal interface parameters for categorical values
    print(f"\nMost Common Daily Activity (Mode)   : {df['Primary_Activity'].mode()[0]}")
    print(f"Median of Student Productivity Rating: {df['Productivity_Rating'].median()}")
    print("="*65)

    # ==========================================================
    # 3. INFERENTIAL STATISTICS (Core Hypothesis Tests Engine)
    # ==========================================================
    print("\n--- INFERENTIAL STATISTICS RESULTS ---")

    # A. Pearson Correlation Test: Screen Time vs Sleep Duration
    r_sleep, p_sleep = stats.pearsonr(df['Screen_Time_Hours'], df['Sleep_Duration_Hours'])
    print(f"1. Screen Time vs Sleep Duration Correlation (r): {r_sleep:.3f} (p-value: {p_sleep:.5f})")

    # B. Spearman Rank Correlation Test: Screen Time vs Productivity Rating
    r_prod, p_prod = stats.spearmanr(df['Screen_Time_Hours'], df['Productivity_Rating'])
    print(f"2. Screen Time vs Productivity Spearman Rank (r): {r_prod:.3f} (p-value: {p_prod:.5f})")

    # C. Chi-Square Test of Independence: Gender vs Primary Activity Focus
    contingency_table = pd.crosstab(df['Gender'], df['Primary_Activity'])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n3. Chi-Square Test (Gender vs App Interface Preference):")
    print(f"   Chi2 Statistic: {chi2:.3f}, P-Value Result: {p_chi2:.5f}")
    print("="*65)

    # ==========================================================
    # 4. DATA VISUALIZATION DASHBOARD GENERATION
    # ==========================================================
    # Setting configuration for layout matrices 
    plt.figure(figsize=(16, 10))

    # Plot 1: Scatter Plot with Linear Regression Fit Line
    plt.subplot(2, 2, 1)
    sns.regplot(data=df, x='Screen_Time_Hours', y='Sleep_Duration_Hours', 
                scatter_kws={'alpha':0.6, 'color':'purple'}, line_kws={'color':'red', 'linewidth':2})
    plt.title('Impact of Screen Time on Sleep Duration', fontsize=12, fontweight='bold')
    plt.xlabel('Daily Screen Time (Hours)')
    plt.ylabel('Sleep Duration (Hours)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Box Plot for Platform Distributions vs Screen Time Focus
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Primary_Activity', y='Screen_Time_Hours', palette='Set2')
    plt.title('Screen Time Distribution Across Activities', fontsize=12, fontweight='bold')
    plt.xlabel('Primary Activity Focus')
    plt.ylabel('Screen Time (Hours)')

    # Plot 3: Bar Chart tracking Productivity Scale Trends
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='Productivity_Rating', y='Screen_Time_Hours', errorbar=None, palette='Blues_r')
    plt.title('Average Screen Time per Productivity Score', fontsize=12, fontweight='bold')
    plt.xlabel('Productivity Rating (1=Low, 5=High)')
    plt.ylabel('Avg Screen Time (Hours)')

    # Plot 4: Clustered Count Plot for Multi-Variable Groupings
    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='Primary_Activity', hue='Gender', palette='pastel')
    plt.title('App Category Preferences Sorted by Gender', fontsize=12, fontweight='bold')
    plt.xlabel('Primary Activity Type')
    plt.ylabel('Count of Users')
    plt.xticks(rotation=10)

    # Rendering clean structural outputs layout
    plt.tight_layout()
    plt.show()

    from sklearn.linear_model import LinearRegression

# Training a simple prediction model
X = df[['Screen_Time_Hours']]
y = df['Sleep_Duration_Hours']
model = LinearRegression().fit(X, y)

# Predict for a custom screen time (e.g., 8 hours)
custom_screen_time = [[8.0]]
predicted_sleep = model.predict(custom_screen_time)
print(f"🔮 ML Model Prediction: For 8 hours of screen time, predicted sleep is {predicted_sleep[0]:.1f} hours.")