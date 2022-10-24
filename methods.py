import lib as lb

#loading data
@lb.st.cache
def load_data(data_url):
    data = lb.pd.read_json('data/'+data_url)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

#to display bar diagram
def plot_bar(X,Y,x_label,y_label,clr):
    fig , ax = lb.plt.subplots()
    lb.plt.bar(X , Y ,color=clr)
    # plt.rcParams["figure.figsize"] = (15,5)
    lb.plt.xlabel(x_label)
    lb.plt.ylabel(y_label)
    # plt.title(title)
    lb.st.pyplot(fig)

# a function which display a progress bar while loading the data
def loader(data_url):
    # upgrading the state automaticaly
    data_load_state = lb.st.empty()
    bar = lb.st.progress(0)
    for i in range(100):
        data_load_state.text(f'Loading data... {i+1}➗')
        bar.progress(i + 1)
        lb.time.sleep(0.1)
    df = load_data(data_url)
    data_load_state.text('Loading Over 💯➗!!')
    return df
def display_countplot(data,title):
    fig , ax = lb.plt.subplots()
    lb.sns.countplot(data)
    lb.plt.title(title)
    lb.st.pyplot(fig)