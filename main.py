import lib as lb
import methods as mt

@lb.st.cache
def readme():
    with open('readme.md', 'r') as f:
        output = f.read()
    return output

def main():
    lb.st.title("PFIZER DIGITAL FOOTPRINT (PDF) EVALUATION")
    lb.st.info("Welcome")
    lb.st.sidebar.title('*SELECT THE SOURCES*')
    lb.st.sidebar.header('Differents places where we evaluate the digital impact of the company Pfizer')
    
    #select box
    page = lb.st.sidebar.selectbox(
                          "Select data to visualize",
                          [
                            "On news paper (from Google News)",
                            "Twitter (User's tweets)",
                            "Readme",
                          ],)
    #Readme page
    if page == "Readme":
        with lb.st.container():
            lb.st.balloons()
            lb.st.markdown(readme())
            
    #twitter data
    elif page == "Twitter (User's tweets)":
        
        data_url = 'tt.json'
        tt = mt.loader(data_url)

        lb.st.write("Let's take a look to our data scrapped data from twitter")
        tt = tt[['date','content','hashtags','lang']]
        with lb.st.container():
            lb.st.write(tt)
            lb.st.write("we took ",tt.shape[0]," random tweets")
            lb.st.write("the languages are distributed as follows :")
            mt.display_countplot(tt.lang,"Tweets languages repartition")

    #Google news data
    elif page == "On news paper (from Google News)":
        pass

main()