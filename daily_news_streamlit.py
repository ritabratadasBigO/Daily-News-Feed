import pickle
import streamlit as st
import os
import requests
import logging
import tqdm
from datetime import datetime
import base64

# Define the function to download the image
def download_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logging.error(f"Error downloading image {url}: {e}")
        return None

# Define the function to save the image locally
def save_image_locally(image_data, image_url):
    image_name = os.path.basename(image_url)
    image_path = os.path.join('temp_images', image_name)
    os.makedirs('temp_images', exist_ok=True)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    return image_path

# Load the data dictionary
with open('data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)

st.set_page_config(layout="wide")
# # Inject custom CSS to set the background color to white
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Initiate the streamlit app with a title
# # Download the inShorts logo from the net
# logo_url = 'https://dhurin.in/wp-content/uploads/2022/06/logo-dhurin.png'
# logo_data = download_image(logo_url)
# logo_path = save_image_locally(logo_data, logo_url)

# # Place the inShorts logo at the top of the page and resize it to 20 pixels
# st.image(logo_path,width=10,use_column_width=True)

# # Display the data in a Streamlit app like the inShorts website
# # Set the title of the page to 'inShorts News' and align it to the center
# st.markdown("<h1 style='text-align: center;'>Today's Top New Articles</h1>", unsafe_allow_html=True)
# st.markdown("---")

# # Display the news articles
# for article in data_dict:
#     col1,col2 = st.columns(2)
#     # Embed the article link as hyperlink in the title and keep the title font size to 20
#     title = f"<a href='{article['Article_link']}' style='font-size:25px;'>{article['Title']}</a>"
#     # st.markdown(title, unsafe_allow_html=True)
#     with col2:
#         st.write(title, unsafe_allow_html=True)

#     # Attach the image for the corresponding article link to the left of the title and set the image width and height to 300 by 200 pixels
#     if article.get('Image_URL'):
#         try:
#             # Download the image from the url and attach it to the title
#             image_data = download_image(article['Image_URL'])

#             if image_data:
#                 image_path = save_image_locally(image_data, article['Image_URL'])
#                 # # Create an HTML block with the image floated to the left
#                 # image_html = f"""
#                 # <div style='display: flex; align-items: center;'>
#                 #     <img src='data:image/png;base64,{image_path.encode("base64").decode()}' style='width: 50px; height: 50px; margin-right: 5px;' />
#                 #     <div>{title}</div>
#                 # </div>
#                 # """
#                 # st.markdown(image_html, unsafe_allow_html=True)

#                 # Align the image to the left of where the title starts 
#                 with col1:
#                     st.image(image_path,width=200,use_column_width=True)
                



#                 # st.image(image_path, caption=article['Title'], use_column_width=False, width=200, output_format='auto')
#                 os.remove(image_path)
#             else:
#                 st.write("Image not found.")
#         except Exception as e:
#             st.write("Image not found.")
#     else:
#         st.write("Image not found.")

#     # Display the published date of the article below the article title and set the font size to 12 and make it bold
#     with col2:
#         pub_date = f"<p style='font-size:15px;font-weight:bold;'>Published Date: {article['Published'].split()[0]}</p>"
#         st.write(pub_date, unsafe_allow_html=True)
    
#     # Display the summary of the article just below the title and set the font size to 16 and style to New Times Roman
#     with col2:
#         summ = f"<p style='font-size:14px;font-family:Times New Roman;'>{article['Send Summary']}</p>"
#         st.markdown(summ, unsafe_allow_html=True)
#     # Display a horizontal rule to separate the articles
#     st.markdown("---")

# # Display the inshorts logo at the bottom of the page and set the width to 20 pixels
# st.image(logo_path, width=20,use_column_width=True)

# Load the data dictionary
with open('data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Inject custom CSS to set the background color to white
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    .article-box {
        border: 5px solid pink;
        padding: 10px;
        margin-bottom: 5px;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
        border-radius: 10px; /* Adds rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Initiate the streamlit app with a title
# Download the inShorts logo from the net
logo_url = 'https://dhurin.in/wp-content/uploads/2022/06/logo-dhurin.png'
logo_data = download_image(logo_url)
encoded_logo = base64.b64encode(logo_data).decode()
# Center align the logo
logo_html = f"<img src='data:image/png;base64,{encoded_logo}' style='width:200px; display: block; margin-left: auto; margin-right: auto;' />"

# Place the inShorts logo at the top of the page
st.markdown(logo_html, unsafe_allow_html=True)

# Set the title of the page to 'Today's Top New Articles' and align it to the center
st.markdown("<h1 style='text-align: center;'>Daily News Feed</h1>", unsafe_allow_html=True)


# Display the news articles
for article in data_dict:
    # Start constructing the HTML content
    html_content = "<div class='article-box'>"
    html_content += "<table style='width:100%;'><tr>"

    # Image column
    html_content += "<td style='width:10%;'>"
    if article.get('Image_URL'):
        try:
            image_data = download_image(article['Image_URL'])
            if image_data:
                # Encode image to base64
                encoded_image = base64.b64encode(image_data).decode()
                # Create image tag
                html_content += f"<img src='data:image/jpeg;base64,{encoded_image}' style='width:150px; height:90px;' />"
            else:
                html_content += "Image not found."
        except Exception as e:
            html_content += "Image not found."
    else:
        html_content += "Image not found."
    html_content += "</td>"

    # Text column
    html_content += "<td style='vertical-align:top;'>"
    # Title
    title = f"<a href='{article['Article_link']}' style='font-size:25px;'>{article['Title']}</a>"
    html_content += title
    # Published date
    pub_date = f"<p style='font-size:12px;font-weight:bold;'>Published Date: {article['Published'].split()[0]}</p>"
    html_content += pub_date
    # Summary
    summ = f"<p style='font-size:18px;font-family:Times New Roman;'>{article['Send Summary']}</p>"
    html_content += summ
    html_content += "</td>"

    html_content += "</tr></table>"
    html_content += "</div>"

    # Display the HTML content
    st.markdown(html_content, unsafe_allow_html=True)

# Display the inShorts logo at the bottom of the page
st.markdown(logo_html, unsafe_allow_html=True)




