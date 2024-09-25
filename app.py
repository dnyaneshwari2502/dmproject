
import streamlit as st

# Set page config
st.set_page_config(page_title="Sea Level Rise Analysis", page_icon="🌊", layout="wide")

# Function to add custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f5;  
            color: #333;                
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;  
        }
        h1, h2, h3, h4 {
            font-family: 'Arial', sans-serif;  
            font-weight: bold;                  
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()


# Sidebar for Navigation
st.sidebar.title("Navigation")
# Main sections
section = st.sidebar.selectbox("Select a Section", ["Introduction", "EDA", "Models Implemented", "Conclusion", "Team"])

# if section == "Introduction":
#     st.header("Introduction")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image("images/example_image.jpg", caption="Example Image", width=150)
    
#     with col2:
#         st.write("The primary causes of sea level rise associated with climate change are...")


# Title of the web app
st.title("Data Science Project")


# Button Navigation
# section = st.sidebar.radio("Go to", ("Introduction", "EDA", "Models Implemented", "Conclusion", "Team"))

# Introduction Section
if section == "Introduction":
    st.header("Introduction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("image1.jpeg", caption="Example Image", width=150)
    
    with col2:
        st.write("The primary causes of sea level rise associated with climate change are...")

    st.markdown("""
    **The Nature of the Topic**  
    The primary causes of sea level rise associated with climate change are the thermal expansion of seawater and the melting of ice from glaciers and polar regions. Coastal erosion, seawater intrusion, habitat loss, and community displacement are some of its effects. It may result in major harm to infrastructure and higher adoption costs from an economic perspective.

    In order to analyze trends and forecast the future, data analysis techniques like linear regression are used to inform efficient planning and policy.  
    **The major goal is to analyze the trend in sea level rise and make future predictions using linear regression.**

    **It considers several key aspects**:
    - **A. Data-Driven Analysis**: Focuses on the usage of quantitative methods to investigate sea data and recognize tendencies over time.
    - **B. Employs statistical techniques like linear regression**, a statistical technique to identify relationships among variables.
    - **C. It basically addresses the consequence of climate change**, which indirectly helps to quantify the rise in sea levels.
    - **D. This project aims to not only analyze the historical data but also make predictions about the future sea level scenarios**.
    - **E. This idea combines elements of statistics, geography, environmental sciences, and social sciences**, making it a multidisciplinary topic.

    **Why it is important**:  
    The Analysis of Sea Level Rise is significant because it directly addresses the ongoing global challenge of climate change, particularly highlighting the increasing rates of sea level rise. Sea levels have been steadily rising due to melting polar ice caps and glaciers, as well as the thermal expansion of seawater caused by global warming.  
    **It has been observed that the global sea levels are rising as a result of human-caused global warming, with recent rates being unprecedented over the past 2,000-plus years.** This phenomenon needs high attention as it threatens coastal ecosystems, human settlements, and economic stability, as more than 40% of the global population lives within 100 kilometers of coastlines.

    **By analyzing historical sea level data**, we aim to provide a detailed understanding of past trends and offer predictive insights for future sea level rise.  
    This project is critical for governments, climate researchers, and policymakers to make informed strategies for mitigating risks associated with flooding, coastal erosion, and saltwater intrusion. **These events can lead to the displacement of communities, loss of biodiversity, and substantial financial costs to coastal infrastructure.**

    This project's data-driven approach will highlight the urgency of adopting climate resilience measures. **The insights generated will aid in designing flood defenses, creating sustainable cities, and advocating for reduced greenhouse gas emissions.** The ability to predict sea level rise trajectories is crucial for international negotiations on climate change policies, helping to quantify risks to vulnerable regions.

    **To conclude**, this project will not only make scientific contributions but also foster societal preparedness and resilience in addressing one of the most pressing environmental issues of the century.

    **What has been done so far and what gaps remain**:  
    - Researched different types of attributes needed for developing data models.
    - Explored various datasets to understand the structure, including year and rise in sea level.
    - Collected datasets and are working on curating them for further analysis.
    - **Major challenge**: Collecting datasets for different locations.

    **Who are affected**:
    - **People residing in low-lying coastal regions** face displacement and loss of property.
    - **Tourism, real estate, and fishing industries** may be affected by flooding and erosion, leading to significant losses.
    - **Cities with buildings, utilities, and roads along the shore** are at risk of flooding.
    """)


# EDA Section
elif section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("""
    In this section, we will perform data cleaning, visualization, and analysis to understand the patterns in the dataset.
    """)

    # Example plot (for visualization)
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Example dataframe
    data = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50]
    })

    st.line_chart(data)

# Models Implemented Section
elif section == "Models Implemented":
    st.header("Models Implemented")
    st.write("""
    In this section, we showcase the different machine learning models implemented to solve the problem at hand.
    """)

    # Example model explanation
    st.subheader("Model 1: Linear Regression")
    st.write("Linear regression was used as the first model to predict the target variable...")

    st.subheader("Model 2: Random Forest")
    st.write("Random Forest was chosen due to its ability to handle non-linear data and offer better accuracy...")

# Conclusion Section
elif section == "Conclusion":
    st.header("Conclusion")
    st.write("""
    In conclusion, the models implemented performed well, with Random Forest providing the best accuracy for the dataset.
    """)

# Team Section
# elif section == "Team":
#     st.header("Team")
#     st.write("""
#     This project was built by a dedicated team of data scientists.
#     """)

#     st.subheader("Team Members")
#     st.write("1. Varun Reddy Mamidala")
#     st.write("2. Dnyaneshwari Rakshe")
#     st.write("3. Tanvi Nimbalkar")
# Team Section
# elif section == "Team":
#     st.header("Team")
#     st.write("""
#     This project was built by a dedicated team of data scientists.
#     """)

#     st.subheader("Team Members")

#     # Display teammate 1 with photo
#     st.write("1. Varun Reddy Mamidala")
#     st.image("image1.jpeg", caption="Varun Reddy Mamidala", width=150)  # Add Varun's photo

#     # Display teammate 2 with photo
#     st.write("2. Dnyaneshwari Rakshe")
#     st.image("image2.jpeg", caption="Dnyaneshwari Rakshe", width=150) 

#     # Display teammate 3 with photo
#     st.write("3. Tanvi Nimbalk")
#     st.image("image3.jpeg", caption="Tanvi Nimbalkar", width=150) 

elif section == "Team":
    st.header("Team")
    st.write("""
    This project was built by a dedicated team of data scientists.
    """)

    st.subheader("Team Members")

    # Create 3 columns for the team members
    col1, col2, col3 = st.columns(3)

    # Display teammate 1 in the first column
    with col1:
        st.write("1. Varun Reddy Mamidala")
        st.image("image1.jpeg", caption="Varun Reddy Mamidala", width=150)

    # Display teammate 2 in the second column
    with col2:
        st.write("2. Dnyaneshwari Rakshe")
        st.image("image2.jpeg", caption="Dnyaneshwari Rakshe", width=150)

    # Display teammate 3 in the third column
    with col3:
        st.write("3. Tanvi Nimbalkar")
        st.image("image3.jpeg", caption="Tanvi Nimbalkar", width=150)
