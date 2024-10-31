
# # import streamlit as st

# # # Set page config
# # st.set_page_config(page_title="Sea Level Rise Analysis", page_icon="🌊", layout="wide")

# # # Function to add custom CSS
# # def add_custom_css():
# #     st.markdown()
# #         """
# #         <style>
# #         body {
# #             background-color: #f0f2f5;  
# #             color: #333;                
# #         }
# #         .sidebar .sidebar-content {
# #             background-color: #ffffff;  
# #         }
# #         h1, h2, h3, h4 {
# #             font-family: 'Arial', sans-serif;  
# #             font-weight: bold;                  
# #         }
# #         </style>
# #         """,
# #         unsafe_allow_html=True
# #     )

# # add_custom_css()


# # # Sidebar for Navigation
# # st.sidebar.title("Navigation")
# # # Main sections
# # section = st.sidebar.selectbox("Select a Section", ["Introduction", "EDA", "Models Implemented", "Conclusion", "Team"])

# # # if section == "Introduction":
# # #     st.header("Introduction")
# # #     col1, col2 = st.columns(2)
    
# # #     with col1:
# # #         st.image("images/example_image.jpg", caption="Example Image", width=150)
    
# # #     with col2:
# # #         st.write("The primary causes of sea level rise associated with climate change are...")


# # # Title of the web app
# # st.title("Data Science Project")


# # # Button Navigation
# # # section = st.sidebar.radio("Go to", ("Introduction", "EDA", "Models Implemented", "Conclusion", "Team"))

# # # Introduction Section
# # if section == "Introduction":
# #     st.header("Introduction")
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.image("image1.jpeg", caption="Example Image", width=150)
    
# #     with col2:
# #         st.write("The primary causes of sea level rise associated with climate change are...")

# #     st.markdown("""
# #     **The Nature of the Topic**  
# #     The primary causes of sea level rise associated with climate change are the thermal expansion of seawater and the melting of ice from glaciers and polar regions. Coastal erosion, seawater intrusion, habitat loss, and community displacement are some of its effects. It may result in major harm to infrastructure and higher adoption costs from an economic perspective.

# #     In order to analyze trends and forecast the future, data analysis techniques like linear regression are used to inform efficient planning and policy.  
# #     **The major goal is to analyze the trend in sea level rise and make future predictions using linear regression.**

# #     **It considers several key aspects**:
# #     - **A. Data-Driven Analysis**: Focuses on the usage of quantitative methods to investigate sea data and recognize tendencies over time.
# #     - **B. Employs statistical techniques like linear regression**, a statistical technique to identify relationships among variables.
# #     - **C. It basically addresses the consequence of climate change**, which indirectly helps to quantify the rise in sea levels.
# #     - **D. This project aims to not only analyze the historical data but also make predictions about the future sea level scenarios**.
# #     - **E. This idea combines elements of statistics, geography, environmental sciences, and social sciences**, making it a multidisciplinary topic.

# #     **Why it is important**:  
# #     The Analysis of Sea Level Rise is significant because it directly addresses the ongoing global challenge of climate change, particularly highlighting the increasing rates of sea level rise. Sea levels have been steadily rising due to melting polar ice caps and glaciers, as well as the thermal expansion of seawater caused by global warming.  
# #     **It has been observed that the global sea levels are rising as a result of human-caused global warming, with recent rates being unprecedented over the past 2,000-plus years.** This phenomenon needs high attention as it threatens coastal ecosystems, human settlements, and economic stability, as more than 40% of the global population lives within 100 kilometers of coastlines.

# #     **By analyzing historical sea level data**, we aim to provide a detailed understanding of past trends and offer predictive insights for future sea level rise.  
# #     This project is critical for governments, climate researchers, and policymakers to make informed strategies for mitigating risks associated with flooding, coastal erosion, and saltwater intrusion. **These events can lead to the displacement of communities, loss of biodiversity, and substantial financial costs to coastal infrastructure.**

# #     This project's data-driven approach will highlight the urgency of adopting climate resilience measures. **The insights generated will aid in designing flood defenses, creating sustainable cities, and advocating for reduced greenhouse gas emissions.** The ability to predict sea level rise trajectories is crucial for international negotiations on climate change policies, helping to quantify risks to vulnerable regions.

# #     **To conclude**, this project will not only make scientific contributions but also foster societal preparedness and resilience in addressing one of the most pressing environmental issues of the century.

# #     **What has been done so far and what gaps remain**:  
# #     - Researched different types of attributes needed for developing data models.
# #     - Explored various datasets to understand the structure, including year and rise in sea level.
# #     - Collected datasets and are working on curating them for further analysis.
# #     - **Major challenge**: Collecting datasets for different locations.

# #     **Who are affected**:
# #     - **People residing in low-lying coastal regions** face displacement and loss of property.
# #     - **Tourism, real estate, and fishing industries** may be affected by flooding and erosion, leading to significant losses.
# #     - **Cities with buildings, utilities, and roads along the shore** are at risk of flooding.
# #     """)


# # # EDA Section
# # elif section == "EDA":
# #     st.header("Exploratory Data Analysis (EDA)")
# #     st.write("""
# #     In this section, we will perform data cleaning, visualization, and analysis to understand the patterns in the dataset.
# #     """)

# #     # Example plot (for visualization)
# #     import pandas as pd
# #     import matplotlib.pyplot as plt
# #     import seaborn as sns

# #     # Example dataframe
# #     data = pd.DataFrame({
# #         "x": [1, 2, 3, 4, 5],
# #         "y": [10, 20, 30, 40, 50]
# #     })

# #     st.line_chart(data)

# # # Models Implemented Section
# # elif section == "Models Implemented":
# #     st.header("Models Implemented")
# #     st.write("""
# #     In this section, we showcase the different machine learning models implemented to solve the problem at hand.
# #     """)

# #     # Example model explanation
# #     st.subheader("Model 1: Linear Regression")
# #     st.write("Linear regression was used as the first model to predict the target variable...")

# #     st.subheader("Model 2: Random Forest")
# #     st.write("Random Forest was chosen due to its ability to handle non-linear data and offer better accuracy...")

# # # Conclusion Section
# # elif section == "Conclusion":
# #     st.header("Conclusion")
# #     st.write("""
# #     In conclusion, the models implemented performed well, with Random Forest providing the best accuracy for the dataset.
# #     """)

# # # Team Section
# # # elif section == "Team":
# # #     st.header("Team")
# # #     st.write("""
# # #     This project was built by a dedicated team of data scientists.
# # #     """)

# # #     st.subheader("Team Members")
# # #     st.write("1. Varun Reddy Mamidala")
# # #     st.write("2. Dnyaneshwari Rakshe")
# # #     st.write("3. Tanvi Nimbalkar")
# # # Team Section
# # # elif section == "Team":
# # #     st.header("Team")
# # #     st.write("""
# # #     This project was built by a dedicated team of data scientists.
# # #     """)

# # #     st.subheader("Team Members")

# # #     # Display teammate 1 with photo
# # #     st.write("1. Varun Reddy Mamidala")
# # #     st.image("image1.jpeg", caption="Varun Reddy Mamidala", width=150)  # Add Varun's photo

# # #     # Display teammate 2 with photo
# # #     st.write("2. Dnyaneshwari Rakshe")
# # #     st.image("image2.jpeg", caption="Dnyaneshwari Rakshe", width=150) 

# # #     # Display teammate 3 with photo
# # #     st.write("3. Tanvi Nimbalk")
# # #     st.image("image3.jpeg", caption="Tanvi Nimbalkar", width=150) 

# # elif section == "Team":
# #     st.header("Team")
# #     st.write("""
# #     This project was built by a dedicated team of data scientists.
# #     """)

# #     st.subheader("Team Members")

# #     # Create 3 columns for the team members
# #     col1, col2, col3 = st.columns(3)

# #     # Display teammate 1 in the first column
# #     with col1:
# #         st.write("1. Varun Reddy Mamidala")
# #         st.image("image1.jpeg", caption="Varun Reddy Mamidala", width=150)

# #     # Display teammate 2 in the second column
# #     with col2:
# #         st.write("2. Dnyaneshwari Rakshe")
# #         st.image("image2.jpeg", caption="Dnyaneshwari Rakshe", width=150)

# #     # Display teammate 3 in the third column
# #     with col3:
# #         st.write("3. Tanvi Nimbalkar")
# #         st.image("image3.jpeg", caption="Tanvi Nimbalkar", width=150)
# import streamlit as st
# import pandas as pd

# # Set page config
# st.set_page_config(page_title="Sea Level Rise Analysis", page_icon="🌊", layout="wide")

# # Function to add custom CSS
# def add_custom_css():
#     st.markdown(
#         """
#         <style>
#         body {
#             background-color: #eaf1f9;  
#             font-family: 'Arial', sans-serif; 
#             color: #333;                
#         }
#         .sidebar .sidebar-content {
#             background-color: #ffffff;  
#             border-right: 2px solid #0072b8; 
#         }
#         h1, h2, h3, h4 {
#             color: #0072b8; 
#             font-weight: bold;                  
#         }
#         .stButton {
#             background-color: #0072b8; 
#             color: white; 
#             border-radius: 5px; 
#             padding: 10px; 
#         }
#         .stButton:hover {
#             background-color: #005f8a; 
#         }
#         .custom-section {
#             border: 1px solid #0072b8; 
#             border-radius: 10px; 
#             padding: 20px; 
#             background-color: white; 
#             margin-bottom: 20px; 
#         }
#         .hover-image {
#             transition: transform 0.2s; 
#         }
#         .hover-image:hover {
#             transform: scale(1.1); 
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# add_custom_css()

# def collapsible_section(title, content):
#     with st.expander(title):
#         st.write(content)

# # Sidebar for Navigation
# st.sidebar.title("Navigation")
# section = st.sidebar.selectbox("Select a Section", ["Introduction", "Data Cleaning and Preprocessing", "EDA", "Models Implemented", "Conclusion", "Team"], key="nav")

# # Title of the web app
# st.title("Data Science Project")

# # Introduction Section
# if section == "Introduction":
#     st.header("Introduction")
    
#     # Collapsible Q&A format for introduction
#     collapsible_section("What are the primary causes of sea level rise?", """
#     The primary causes of sea level rise associated with climate change are the thermal expansion of seawater and the melting of ice from glaciers and polar regions.
#     """)

#     collapsible_section("What are the effects of sea level rise?", """
#     Coastal erosion, seawater intrusion, habitat loss, and community displacement are some of its effects. It may result in major harm to infrastructure and higher adoption costs from an economic perspective.
#     """)

#     collapsible_section("Why is this analysis important?", """
#     The Analysis of Sea Level Rise is significant because it directly addresses the ongoing global challenge of climate change, particularly highlighting the increasing rates of sea level rise. More than 40% of the global population lives within 100 kilometers of coastlines, making it crucial to understand the implications.
#     """)

#     collapsible_section("What has been done so far?", """
#     - Researched different types of attributes needed for developing data models.
#     - Explored various datasets to understand the structure, including year and rise in sea level.
#     - Collected datasets and are working on curating them for further analysis.
#     - Major challenge: Collecting datasets for different locations.
#     """)

#     # Image section
#     st.image("image1.jpeg", caption="Example Image", width=150, use_column_width='auto', class_='hover-image')

# # Data Cleaning and Preprocessing Section
# elif section == "Data Cleaning and Preprocessing":
#     st.header("Data Cleaning and Preprocessing")
#     st.write("""
#     In this section, we perform data cleaning and preprocessing to prepare the dataset for further analysis and modeling.
#     """)

#     # Example steps of data cleaning
#     collapsible_section("Step 1: Handling Missing Values", """
#     - Checked for missing values and applied appropriate methods such as imputation or removal of rows with missing data.
#     """)

#     collapsible_section("Step 2: Data Normalization", """
#     - Normalized numerical features to ensure they are on the same scale, using techniques like Min-Max scaling or Z-score normalization.
#     """)

#     collapsible_section("Step 3: Feature Engineering", """
#     - Created new features from existing data to improve model performance, such as calculating the rate of sea level rise over time.
#     """)

#     collapsible_section("Step 4: Data Transformation", """
#     - Applied transformations to skewed data distributions and ensured categorical variables were properly encoded.
#     """)

# # EDA Section
# elif section == "EDA":
#     st.header("Exploratory Data Analysis (EDA)")
#     st.write("""
#     In this section, we will perform data cleaning, visualization, and analysis to understand the patterns in the dataset.
#     """)

#     # Example plot (for visualization)
#     data = pd.DataFrame({
#         "x": [1, 2, 3, 4, 5],
#         "y": [10, 20, 30, 40, 50]
#     })

#     st.line_chart(data)

# # Models Implemented Section
# elif section == "Models Implemented":
#     st.header("Models Implemented")
#     st.write("""
#     In this section, we showcase the different machine learning models implemented to solve the problem at hand.
#     """)

#     # Example model explanation
#     st.subheader("Model 1: Linear Regression")
#     st.write("Linear regression was used as the first model to predict the target variable...")

#     st.subheader("Model 2: Random Forest")
#     st.write("Random Forest was chosen due to its ability to handle non-linear data and offer better accuracy...")

# # Conclusion Section
# elif section == "Conclusion":
#     st.header("Conclusion")
#     st.write("""
#     In conclusion, the models implemented performed well, with Random Forest providing the best accuracy for the dataset.
#     """)

# # Team Section
# elif section == "Team":
#     st.header("Team")
#     st.write("""
#     This project was built by a dedicated team of data scientists.
#     """)

#     st.subheader("Team Members")
#     st.write("1. Varun Reddy Mamidala")
#     st.image("image1.jpeg", caption="Varun Reddy Mamidala", width=150)  # Add Varun's photo

#     st.write("2. Dnyaneshwari Rakshe")
#     st.image("image2.jpeg", caption="Dnyaneshwari Rakshe", width=150) 

#     st.write("3. Tanvi Nimbalkar")
#     st.image("image3.jpeg", caption="Tanvi Nimbalkar", width=150)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns  # Import for seaborn visualizations


# Set page config
st.set_page_config(page_title="Sea Level Rise Analysis", page_icon="🌊", layout="wide")

# Function to add custom CSS
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #eaf1f9;  
            font-family: 'Arial', sans-serif; 
            color: #333;                
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;  
            border-right: 2px solid #0072b8; 
        }
        h1, h2, h3, h4 {
            color: #0072b8; 
            font-weight: bold;                  
        }
        .stButton {
            background-color: #0072b8; 
            color: white; 
            border-radius: 5px; 
            padding: 10px; 
        }
        .stButton:hover {
            background-color: #005f8a; 
        }
        .custom-section {
            border: 1px solid #0072b8; 
            border-radius: 10px; 
            padding: 20px; 
            background-color: white; 
            margin-bottom: 20px; 
        }
        img:hover {
            transform: scale(1.1); 
            transition: transform 0.2s;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

def collapsible_section(title, content):
    with st.expander(title):
        st.write(content)

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Select a Section", ["Introduction", "Data Collection and Cleaning", "Data Visualizations", "Models Implemented", "Conclusion", "Team"], key="nav")

# Title of the web app
st.title("Data Science Project")

# Introduction Section
if section == "Introduction":
    st.header("Introduction")
    
    # Collapsible Q&A format for introduction
    collapsible_section("What are the primary causes of sea level rise?", """
    The primary causes of sea level rise linked to climate change are the thermal expansion of seawater and the melting of ice from glaciers and polar regions. As global temperatures increase, seawater warms and expands, taking up more space, which contributes significantly to the rising levels observed in oceans worldwide. This phenomenon, known as thermal expansion, accounts for about half of the observed sea level rise in recent decades. Additionally, higher temperatures accelerate the melting of glaciers and ice sheets in polar regions like Greenland and Antarctica. When ice from these land-based sources melts, it flows into the ocean, directly increasing sea levels. This melting is particularly concerning because ice sheets hold vast amounts of water; even small increases in melt rates can lead to considerable changes in sea levels over time. Seasonal changes, particularly in the Arctic, have also shown that ice is melting at an unprecedented rate, affecting habitats and increasing the flow of freshwater into the sea. Rising sea levels have widespread impacts on coastal ecosystems, contributing to shoreline erosion, saltwater intrusion into freshwater sources, and the loss of habitat for marine life. Combined, thermal expansion and ice melt present a complex challenge, as they not only contribute to rising waters but also indicate ongoing changes in Earth’s climate systems. Addressing these root causes is essential to managing and potentially mitigating future sea level rise.
    """)

    collapsible_section("What are the effects of sea level rise?", """
    Sea level rise has a range of significant effects on both natural environments and human communities. Coastal erosion is one of the most visible impacts, as rising waters gradually wear away shorelines, threatening properties and natural habitats. Seawater intrusion is another serious concern, where saltwater encroaches into freshwater aquifers, jeopardizing drinking water supplies and agricultural lands. Habitat loss occurs as rising seas submerge coastal wetlands, mangroves, and estuaries, which serve as critical ecosystems for diverse marine and bird species. This environmental change disrupts ecosystems, leading to declines in biodiversity and impacting food chains.

    For communities near coastlines, rising sea levels can lead to forced relocation and displacement as homes, schools, and businesses are threatened or submerged. This creates significant social challenges, as entire communities may need to move, severing historical ties to land and culture. From an economic standpoint, sea level rise leads to increased costs for adapting infrastructure, such as building seawalls, improving drainage systems, and elevating structures. It also heightens the risk of major damage to roads, bridges, ports, and utilities during extreme weather events. In total, the economic and social burdens of sea level rise are vast, and addressing them requires proactive planning and adaptation to protect vulnerable populations and ecosystems.    """)

    collapsible_section("Why is this analysis important?", """
    The Analysis of Sea Level Rise is significant because it directly addresses the ongoing global challenge of climate change, particularly highlighting the increasing rates of sea level rise. More than 40% of the global population lives within 100 kilometers of coastlines, making it crucial to understand the implications.
    """)
    

    collapsible_section("What has been done so far?", """
    Our team has collectively researched various attributes needed for developing data models, focusing on identifying the ones that are most crucial for our project. Together, we examined factors that might influence sea level rise, such as temperature changes, glacial melt rates, and atmospheric conditions. We also explored multiple datasets to understand their structures and find the most relevant data points, especially those capturing year-to-year trends in sea levels. As a team, we gathered valuable datasets, which we are now curating to ensure they are ready for analysis. This process has involved extensive filtering, cleaning, and standardizing, as the datasets came from diverse sources with varying levels of detail. A significant challenge we faced was in obtaining location-specific datasets, as sea level data differs across regions, making it essential to merge information from multiple sources while ensuring compatibility. Combining these datasets without losing critical information has been both complex and time-intensive. By working through these challenges together, we aim to build an accurate, comprehensive model that reflects global and regional trends in sea level rise. Our collaborative efforts are setting a strong foundation for valuable insights into how various factors impact sea levels across different locations.
    """)
    
    collapsible_section("Questions which we aim to answer through our project?", """
    1. What is the trend of sea level rise over the past few years in the monitored location?
    2. How do seasonal changes affect water levels in the region?
    3. What are the highest and lowest water levels recorded, and what factors might explain these extremes?
    4. How frequently do extreme high tides (above MHHW) occur, and are they becoming more common?
    5. Is there a noticeable difference between high and low tides (MHW and MLW), and is this difference increasing?
    6. How does the recorded Mean Sea Level (MSL) compare to historical MSL averages?
    7. What is the average tidal range, and does it show any signs of increasing or decreasing?
    8. Are there particular months or seasons where the risk of extreme water levels is highest?
    9. Does the data indicate any unusual water level anomalies, and what might be the cause?
    10. How might future sea level rise impact nearby coastal infrastructure and ecosystems if current trends continue?
    """)

    # Image section without `class_`
    st.image("image1.jpeg", caption="Example Image", width=150, use_column_width='auto')

# Data Collection and Cleaning section
elif section == "Data Collection and Cleaning":
    st.header("Data Collection and Cleaning")
    st.write("""
    In this section, we perform data cleaning and preprocessing to prepare the dataset for further analysis and modeling.
    """)

    # Collapsible section for data collection
    with st.expander("Data Collection"):
        st.write("""
        **Initial Data Scraping:**
        - **Objective:** Identify available stations for data collection.
        - **Method:** Scraped data from a relevant source to obtain a list of stations where data could be collected.
        """)
        st.image("initially scrapped data.png", caption="Initial dataset", width=400, use_column_width='auto')
        st.write("""
        **Detailed Data Scraping:**
        - **Objective:** Collect data from each identified station.
        - **Method:** Used an API to scrape the data from each station, retrieving the specific information needed for the project.
        """)

        # Image inside the collapsible section for data collection
        st.image("dataset_image.jpeg", caption="detailed dataset", width=400, use_column_width='auto')

    # Collapsible section for dataset description and structure with image inside
    with st.expander("Dataset Description and Structure"):
        st.write("""
        **DateTime (GMT):**
        - Description: The date and time when the data was recorded, referenced to the GMT time zone.
        - Format: Likely in a YYYY-MM-DD HH:MM:SS format, with data points spaced according to the interval (monthly, in your case).

        **Highest:**
        - Description: The highest recorded water level during the given period (likely the highest tide or surge).
        - Units: Feet (ft).

        **MHHW (Mean Higher High Water):**
        - Description: The average of the higher of the two daily high tides over a 19-year period.
        - Units: Feet (ft).
        - Use: Helps identify higher tidal ranges.

        **MHW (Mean High Water):**
        - Description: The average of all high water levels over a 19-year period.
        - Units: Feet (ft).
        - Use: Represents the average height of the high tides.

        **MSL (Mean Sea Level):**
        - Description: The average sea level based on observations over a period of time.
        - Units: Feet (ft).
        - Use: Often used as a reference point for various measurements, including vertical land movement and sea level rise.

        **MTL (Mean Tide Level):**
        - Description: The average of Mean High Water (MHW) and Mean Low Water (MLW).
        - Units: Feet (ft).
        - Use: Used as a midpoint between high and low tides.

        **MLW (Mean Low Water):**
        - Description: The average of all the low water levels recorded over a 19-year period.
        - Units: Feet (ft).
        - Use: Represents the typical low tide level.

        **MLLW (Mean Lower Low Water):**
        - Description: The average of the lower of the two daily low tides over a 19-year period.
        - Units: Feet (ft).
        - Use: A tidal datum that serves as a baseline for measuring water depth.

        **Lowest:**
        - Description: The lowest recorded water level during the given period.
        - Units: Feet (ft).

        **Inf:**
        - Description: This could represent flags for additional information about the data or indicate missing or extreme data points.
        """)

        # Image inside the collapsible section for dataset description
        st.image("dataset_image.jpeg", caption="Dataset Snapshot", width=400, use_column_width='auto')

    # Collapsible section for handling missing values
    collapsible_section("Handling Missing Values", """
    - Checked for missing values and imputated with mean of the attribute.
    """)

    # Collapsible section for combining date and time columns
    collapsible_section("Combine Date and Time Column", """
    - Combined two different date and time columns into a single DateTime column.
    """)

# Load your dataset
data = pd.read_csv('station 1611400dataaset.csv', parse_dates=['Date', 'Time (GMT)'])

# Ensure that 'Date' and 'Time (GMT)' columns exist and are properly formatted as strings
data['Date'] = data['Date'].astype(str)
data['Time (GMT)'] = data['Time (GMT)'].astype(str)

# Combine 'Date' and 'Time (GMT)' into a new 'Datetime' column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time (GMT)'], errors='coerce')

# Create a new dataset with the combined 'Datetime' column and drop 'Date' and 'Time (GMT)'
new_dataset = data.drop(columns=['Date', 'Time (GMT)'])

# Streamlit Section for Data Visualizations
# Data Visualizations Section
if section == "Data Visualizations":
    st.header("Data Visualizations")

    # Tidal Levels Over Time
    st.subheader("Tidal Levels Over Time")
    fig1, ax1 = plt.subplots()
    data.plot(x='Datetime', y=['Highest', 'Lowest (ft)'], kind='line', ax=ax1)
    plt.title('Tidal Levels Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Tide Levels (ft)')
    st.pyplot(fig1)

    # Frequency Distribution of Mean Sea Level (ft)
    st.subheader("Frequency Distribution of Mean Sea Level (ft)")
    fig2, ax2 = plt.subplots()
    data['MSL (ft)'].plot(kind='hist', bins=20, color='beige', edgecolor='black', ax=ax2)
    plt.title('Frequency Distribution of Mean Sea Level (ft)')
    plt.xlabel('MSL (ft)')
    plt.ylabel('Frequency')
    st.pyplot(fig2)

    # Highest vs. Lowest Tide
    st.subheader("Highest vs. Lowest Tide")
    fig3, ax3 = plt.subplots()
    data.plot.scatter(x='Highest', y='Lowest (ft)', ax=ax3)
    plt.title('Highest Tide vs Lowest Tide')
    plt.xlabel('Highest Tide (ft)')
    plt.ylabel('Lowest Tide (ft)')
    st.pyplot(fig3)

    # Tidal Components Over Time
    st.subheader("Tidal Components Over Time")
    fig4, ax4 = plt.subplots()
    data.plot(x='Datetime', y=['MHHW (ft)', 'MHW (ft)', 'MLW (ft)'], ax=ax4)
    plt.title('Mean Water Levels Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Mean Water Level (ft)')
    st.pyplot(fig4)

    # # Identifying Cyclic or Seasonal Behavior
    # st.subheader("Identifying Cyclic or Seasonal Behavior")
    # fig5, ax5 = plt.subplots()
    # pd.plotting.autocorrelation_plot(data['Highest'], ax=ax5)
    # plt.title('Autocorrelation Plot of Highest Tide')
    # st.pyplot(fig5)

    # Detecting Seasonality and Trends Over Time
    st.subheader("Detecting Seasonality and Trends Over Time")
    fig6, ax6 = plt.subplots()
    pd.plotting.lag_plot(data['Highest'], ax=ax6)
    plt.title('Lag Plot of Highest Tide')
    st.pyplot(fig6)

    from statsmodels.graphics.tsaplots import plot_acf

#     Identifying Cyclic or Seasonal Behavior
#     st.subheader("Identifying Cyclic or Seasonal Behavior")
#     fig5, ax5 = plt.subplots()
#     plot_acf(data['Highest'], ax=ax5)
#     plt.title('Autocorrelation Plot of Highest Tide')
#     st.pyplot(fig5)
    # Mean Tide Levels by Month
    st.subheader("Mean Tide Levels by Month")
    data['Month'] = data['Datetime'].dt.month
    fig8, ax8 = plt.subplots()
    data.groupby('Month')['MTL (ft)'].mean().plot(kind='bar', ax=ax8)
    plt.title('Mean Tide Level by Month')
    plt.xlabel('Month')
    plt.ylabel('Mean Tide Level (ft)')
    st.pyplot(fig8)

    # Tidal Levels with Seasonal Indicators
    st.subheader("Tidal Levels with Seasonal Indicators")
    data['Season'] = data['Datetime'].dt.month % 12 // 3 + 1
    fig9, ax9 = plt.subplots()
    sns.lineplot(x='Datetime', y='Highest', hue='Season', data=data, ax=ax9)
    plt.title('Highest Tide Over Time by Season')
    plt.xlabel('Datetime')
    plt.ylabel('Highest Tide (ft)')
    st.pyplot(fig9)


# Models Implemented Section
elif section == "Models Implemented":
    st.header("Models Implemented")
    st.write("""
    In this section, we discuss the various models that have been implemented to predict sea level rise.
    """)

    # Linear Regression Model
    st.subheader("Model 1: Linear Regression")
    st.write("""
    A simple linear regression model that predicts sea level rise based on various tidal attributes.
    """)

    # Code for Linear Regression
    st.code("""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Define features and target variable
    X = new_dataset[['MHHW (ft)', 'MHW (ft)', 'MSL (ft)']]  # Features
    y = new_dataset['Highest']  # Target variable (assuming 'Highest' as the target)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model and fit it
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')
    """)

    # Placeholder for additional models
    st.subheader("Model 2: Random Forest Regression")
    st.write("""
    A Random Forest model to capture more complex relationships in the data.
    """)

    # Code for Random Forest Regression
    st.code("""
    from sklearn.ensemble import RandomForestRegressor

    # Create a Random Forest model and fit it
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    st.write(f'Random Forest Mean Squared Error: {rf_mse:.2f}')
    """)

# Team Section
elif section == "Team":
    st.header("Team Members")
    st.write("""
    - Team Member 1: [Insert Name]
    - Team Member 2: [Insert Name]
    - Team Member 3: [Insert Name]
    """)

    # You can also include team roles or contributions
    st.write("""
    ### Team Contributions
    - Team Member 1: Data Cleaning and Preprocessing
    - Team Member 2: Model Implementation and Evaluation
    - Team Member 3: Visualization and Presentation
    """)
