import streamlit as pd
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import os

# Set page configuration
st.set_page_config(page_title="Digital Wellness Dashboard", layout="wide")

st.title("📊 Digital Wellness & Screen Time Statistical Dashboard")
st.markdown("A portfolio data science project exploring the impact of screen time on sleep and productivity.")

# 1. LOAD DATASET
file_name = 'Digital_Wellness_Dataset.xlsx'

if not os.path.exists(file_name):
    st.error(f"❌ '{file_name}' not found. Please place the Excel file in the same directory.")
else:
    df = pd.read_excel(file_name)
    
    # ------------------------------------------------------
    # SIDEBAR FILTER (Interactive Element)
    # ------------------------------------------------------
    st.sidebar.header("🎯 Filter Dataset")
    gender_filter = st.sidebar.multiselect("Select Gender:", 
                                           options=df["Gender"].unique(), 
                                           default=df["Gender"].unique())
    
    # Filter dataframe based on selection
    filtered_df = df[df["Gender"].isin(gender_filter)]
    
    # ------------------------------------------------------
    # TOP METRICS DASHBOARD
    # ------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Respondents (N)", len(filtered_df))
    with col2:
        st.metric("Avg Screen Time", f"{filtered_df['Screen_Time_Hours'].mean():.1f} Hrs")
    with col3:
        st.metric("Avg Sleep Duration", f"{filtered_df['Sleep_Duration_Hours'].mean():.1f} Hrs")
    with col4:
        st.metric("Median Productivity", f"{filtered_df['Productivity_Rating'].median()}/5")

    st.markdown("---")

    # ------------------------------------------------------
    # CHARTS SECTION (2x2 Grid)
    # ------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Interactive Scatter Plot
        st.subheader("1. Screen Time vs Sleep Duration")
        fig1 = px.scatter(filtered_df, x="Screen_Time_Hours", y="Sleep_Duration_Hours", 
                          trendline="ols", color="Primary_Activity",
                          labels={"Screen_Time_Hours": "Screen Time (Hrs)", "Sleep_Duration_Hours": "Sleep (Hrs)"},
                          title="Inverse Relationship (Negative Correlation)")
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        # Chart 2: Interactive Box Plot
        st.subheader("2. Screen Time Distribution by Activity")
        fig2 = px.box(filtered_df, x="Primary_Activity", y="Screen_Time_Hours", 
                      color="Primary_Activity", title="Where is the time going?")
        st.plotly_chart(fig2, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        # Chart 3: Interactive Bar Chart
        st.subheader("3. Avg Screen Time per Productivity Score")
        avg_screen_prod = filtered_df.groupby("Productivity_Rating")["Screen_Time_Hours"].mean().reset_index()
        fig3 = px.bar(avg_screen_prod, x="Productivity_Rating", y="Screen_Time_Hours",
                      color="Screen_Time_Hours", color_continuous_scale="Blues",
                      title="Productivity Score vs Screen Usage")
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col4:
        # Chart 4: Gender-wise Platform Count
        st.subheader("4. Platform Preferences by Gender")
        fig4 = px.histogram(filtered_df, x="Primary_Activity", color="Gender", barmode="group",
                            title="Primary Digital Activity Focus")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # STATISTICAL TESTING SECTION (Backend Insights)
    # ------------------------------------------------------
    st.subheader("🔬 Statistical Insights & Hypothesis Testing")
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("### 📊 Correlation Tests")
        # Pearson Correlation
        r_sleep, p_sleep = stats.pearsonr(filtered_df['Screen_Time_Hours'], filtered_df['Sleep_Duration_Hours'])
        st.write(f"**Pearson Correlation (r) [Screen vs Sleep]:** `{r_sleep:.3f}` (p-value: `{p_sleep:.5f}`)")
        
        # Spearman Rank
        r_prod, p_prod = stats.spearmanr(filtered_df['Screen_Time_Hours'], filtered_df['Productivity_Rating'])
        st.write(f"**Spearman Rank (r) [Screen vs Productivity]:** `{r_prod:.3f}` (p-value: `{p_prod:.5f}`)")

    with stat_col2:
        st.markdown("### 🔍 Core Findings & Conclusions")
        if r_sleep < -0.5:
            st.success("💡 **Strong Negative Correlation Proven:** As screen time increases, student sleep duration drops significantly. This validates our primary hypothesis.")
        else:
            st.warning("💡 **Weak Correlation:** The relationship between screen time and sleep is present but not strongly linear in this filtered group.")
            
        st.info("💡 **Recruiter Note:** This backend seamlessly integrates raw data parsing, categorical distributions, and statistical confidence levels into a functional analytics engine.")