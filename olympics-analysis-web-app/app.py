import streamlit as st
import pandas as pd
import numpy as np
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')


df = preprocessor.preprocess(df,region_df)

st.sidebar.title("Olympics Medal Prediction")
st.sidebar.image('images/olym.png')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally','Overall Analysis','Country-wise Analysis','Athlete wise Analysis','Medal Prediction')
)

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Medal Prediction':
    st.title("Medal Prediction")

    adf = pd.read_csv('athletes.csv')
    del adf['dob'], adf['id'], adf['name']

    adf['nationality']= adf['nationality'].astype(str)
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer.fit(adf.iloc[:,2:4])
    adf.iloc[:, 2:4] = imputer.transform(adf.iloc[:, 2:4])
    X = adf.iloc[:, 0:5].values
    Y = adf.iloc[:, [5,6,7]].values
    X = X.transpose()
    le =LabelEncoder()
    X[0] = le.fit_transform(X[0])
    le = LabelEncoder()
    X[4] = le.fit_transform(X[4])
    le = LabelEncoder()
    X[1] = le.fit_transform(X[1])
    gender = adf['sex'].values
    nationality =adf['nationality'].values
    sports = adf['sport'].values
    cat_gen = dict(zip(gender, X[1]))
    cat_nat = dict(zip(nationality, X[0]))
    cat_spo = dict(zip(sports, X[4]))

    cat_data = [cat_nat, cat_gen, cat_spo]
    X = X.transpose()
    ranForReg = RandomForestRegressor(n_estimators=245, random_state =40, max_depth= 14, min_samples_split=10) 
    ranForReg.fit(X, Y)

    def Inputs(country, gender, height, weight, sport):
        c = cat_data[0].get(country.upper())
        g = cat_data[1].get(gender.lower())
        s = cat_data[2].get(sport.lower())
        h = height
        w = weight
        list_data = [c,g,h,w,s]
        pred = ranForReg.predict([list_data])
    
        #for Gold medals 
        if pred[0][0] < 0.036:
            gold = 0
        elif pred[0][0] > 0.036 and pred[0][0] < 0.12:
            gold = 1
        elif pred[0][0] > 0.12 and pred[0][0] < 0.9:
            gold = 2
        elif pred[0][0] > 0.9:
            gold= '2+'
        #For Silver medal
        if pred[0][1] < 0.3:
            silver = 0
        elif pred[0][1] > 0.3 and pred[0][1] < 0.5:
            silver = 1
        elif pred[0][1] > 0.5:
            silver = '1+'
        #for Bronze medal
        if pred[0][2] < 0.21:
            bronze = 0
        elif pred[0][2] > 0.21 and pred[0][2] < 4.5:
            bronze = 1
        elif pred[0][2] > 4.5:
            bronze = '1+'

        prediction = f'Total Number of Gold Medals:{gold} \nTotal Number of Silver Medals:{silver}\nTotal Number of Bronze Medals:{bronze}'
        return prediction

    nation = st.text_input("Country: ")
    gender = st.text_input("Gender: ")
    height = st.text_input("Height in m: ")
    weight = st.text_input("Weight in Kg: ")
    sports = st.text_input("Sport: ") 
    # Predict function
    emdata = ''
    
    # Button for ptediction
    if st.button('Predict'):
        emdata = Inputs(nation, gender, height, weight, sports)
    
    #Final Output
    st.success(emdata)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df,'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize=(20,20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport = st.selectbox('Select a Sport',sport_list)
    x = helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu == 'Country-wise Analysis':

    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)


    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df,selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(temp_df['Weight'],temp_df['Height'],hue=temp_df['Medal'],style=temp_df['Sex'],s=60)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)


