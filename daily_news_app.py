from pygooglenews import GoogleNews
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import re
import random
from premailer import transform
from PIL import Image,UnidentifiedImageError
import io
from transformers import pipeline
from transformers import AutoTokenizer
import google.generativeai as genai
from groq import Groq
import os
import time
import logging
from warnings import filterwarnings
filterwarnings('ignore')

## Store the email settings in os environment variables
os.environ['SENDER_EMAIL'] = 'ritabrata.das@dhurin.in'
os.environ['SENDER_NAME'] = 'Ritabrata Das'
os.environ['SMTP_SERVER'] = 'smtp.office365.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USERNAME'] = 'ritabrata.das@dhurin.in'
os.environ['SMTP_PASSWORD'] = 'dkcxmlvzjsxcdmjz'
os.environ['RECIPIENTS'] = 'ritabrata.das@dhurin.in'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCuZDCq2Z2SSDAT3mb5uNYJiaaWarTn1Es'
os.environ['GROQ_API_KEY'] = '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'

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the summarizer and set the device to GPU if available else use CPU
summarizer = pipeline('summarization', model='facebook/bart-large-cnn',device=-1) 
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')  # Load the tokenizer


# Initialize a file to store the error logs
logging.basicConfig(filename='error_logs.log', level=logging.ERROR)


## Clean up the article contents in each row
def clean_text(text):
    try:
        text = text.replace('\n', ' ') # Replace newline characters with spaces
        text = text.replace('\r', ' ') # Replace carriage return characters with spaces

        #Remove backspaces, tabs, feed characters and forward slashes
        text = re.sub(r'[\t\b\f\v]+', ' ', text)
        
        ## Remove the special characters
        # text = ' '.join(re.findall(r'\w+', text))
        ## Remove non-ascii characters using re
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        ## Remove non-printable characters using re
        text = re.sub(r'[^ -~]+', ' ', text)
        # Remove decimal numbers from the text
        text = re.sub(r'\d+\.\d+', '', text)

        # Remove mathematical operators from the text
        text = re.sub(r'[+\-*/^]+', '', text)

        # And remove any backspaces, tabs, feed characters , forward slashes and backslashes
        text = text.replace("\\", "")

        return text
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error cleaning text: {text}")
        return text


# # Load more content by clicking on the 'Load More' button
def load_more_content(driver,arg):
    try:
        load_more_button = driver.find_element(By.XPATH, f"//button[contains(text(), '{arg}')]")
        load_more_button.click()
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the f'{arg}' button")
        pass


## Now we  Handle pop-up buttons
def click_button_by_text(button_text,driver):
    try:
        button = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, f"//button[contains(text(), '{button_text}')]"))
        )
        button.click()
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the f'{button_text}' button")
        pass

## Now we handle overlays
def close_overlay(driver):
    # Click on the X button to close the overlay
    try:
        x_button = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, """
                //button[
                    translate(@class, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='x' or
                    text()='X' or
                    translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='close'
                ]
            """))
        )
        x_button.click()
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the X button")
        pass

    ## If pop-up is a modal dialog
    # Click on the close button to close the modal dialog
    try:
        modal = WebDriverWait(driver, 2).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'modal-dialog'))
        )
        driver.switch_to.active_element  # Focus on the modal
        accept_button = modal.find_element(By.XPATH, ".//button[contains(text(), 'Accept')]")
        accept_button.click()
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the Accept button")
        pass

    # Dealing with iframes
    try:
        iframe = driver.find_element(By.XPATH, "//iframe[contains(@src, 'consent')]")
        driver.switch_to.frame(iframe)
        accept_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Accept')]")
        accept_button.click()
        driver.switch_to.default_content()  # Switch back to the main content
    except:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the Accept button")
        pass

# Dealing with multiple close buttons
def close_overlay_new(driver, timeout=2, labels=None):
    if labels is None:
        labels = ['Close', 'Dismiss', 'Cancel']

    # Build the XPath conditions
    xpath_conditions = [
        "@class='x'",
        "@aria-label='Close'"
    ]
    for label in labels:
        condition = f"contains(translate(., 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), '{label.upper()}')"
        xpath_conditions.append(condition)

    xpath_expression = f"//button[{' or '.join(xpath_conditions)}]"

    try:
        close_button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, xpath_expression))
        )
        close_button.click()
    except Exception as e:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error clicking on the close button: {e}")
        pass

def parse(url):
        try:
            # Extract raw HTML
            response = requests.get(url)
            # Pass it to BeautifulSoup for further processing
            soup = BeautifulSoup(response.text, 'lxml')

            # Extract main content (can be adjusted based on the website structure)
            content = soup.find_all(['p', 'div', 'article'])

            # Combine the text from all these elements into one string
            clean_text = ' '.join([element.get_text(strip=True) for element in content])

            return clean_text
        except Exception as e:
            logging.error(f"Error parsing content from URL: {url} is {e}")
            return ''

def get_body_text(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'lxml')
        body_text = soup.body.get_text(' ', strip=True)
        return body_text
    except Exception as e:
        logging.error(f"Error getting body text from URL: {url} is {e}")
        return ''

def get_article_content(url):
    try:
        # Initialize variables
        previous_p_count = 0
        stability_counter = 0
        stability_threshold = 3  # Number of times the counts should remain the same
        max_wait_time = 5  # Maximum time to wait in seconds
        start_time = time.time()  # Start the timer

        driver = webdriver.Chrome() # Start the Chrome browser


        driver.get(url) # Open the URL

        ## Dealing with Overlays
        close_overlay(driver)

        close_overlay_new(driver, labels=['Close', 'Dismiss', 'Cancel', 'No Thanks']) # Close overlays with different labels

        buttons = ['Accept', 'Allow',  'Close', 'Cancel',
                   'I Accept','Not Now','Continue to Site','Ok']

        for text in buttons:
            click_button_by_text(text,driver) # Accept cookies and close pop-ups

       
        # # Load more content by clicking on the 'Load More' or 'Read More' button
        load_more_content(driver,'Load More')
        load_more_content(driver,'Read More')
        

        while stability_counter < stability_threshold:
            # Wait for new elements to load
            time.sleep(1)  # Adjust based on page loading behavior

            # Scroll down to load more content (if applicable)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Get current counts
            current_p_elements = driver.find_elements(By.TAG_NAME, 'p')

            current_p_count = len(current_p_elements)


            # Check if counts have remained the same
            if (current_p_count == previous_p_count):
                stability_counter += 1
            else:
                stability_counter = 0  # Reset counter if counts have changed

            previous_p_count = current_p_count

            # Check for maximum wait time
            if time.time() - start_time > max_wait_time:
                break

        # All elements are assumed to be loaded

        soup = BeautifulSoup(driver.page_source, 'html.parser') # Parse the HTML content of the page


        driver.quit() # Close the browser
        
        # Extract the paragraphs from the article
        paragraphs = soup.find_all('p')


        article_content = ''
        try:
            article_content = '\n'.join([p.get_text() for p in paragraphs]) # Join the text of all paragraphs
        except:
            article_content = '\n'.join([p.text for p in paragraphs]) # Join the text of all paragraphs

        return article_content
            
    except Exception as e:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error scraping content from URL: {url} is {e}")
        return ''

def base_scraper(url):
    try:
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the page content
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = ''
            # Extract and print specific content (like all paragraph texts)
            for paragraph in soup.find_all('p'):
                paragraphs += paragraph.text() + ' '

            return paragraphs
        else:
            return ''
    except Exception as e:
        logging.error(f"Error scraping content from URL: {url} is {e}")
        return ''


# Define the summarization settings
def get_summarization_settings():
    return 6, 200



# Updated function to summarize content using chunking
def summarize_content(content, num_sentences, word_limit):
    try:
        max_input_tokens = 1024  # Max tokens for BART model
        max_summary_tokens = int(word_limit / 0.75)  # Approximate tokens in summary

        # Tokenize content
        content_words = content.split()
        total_words = len(content_words)
        chunks = []

        # Break content into chunks
        for i in range(0, total_words, max_input_tokens):
            chunk = ' '.join(content_words[i:i + max_input_tokens])
            chunks.append(chunk)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_summary_tokens, min_length=5, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        # Combine summaries
        combined_summary = ' '.join(summaries)

        # Post-process to enforce number of sentences and word limit
        sentences = combined_summary.split('. ')
        if len(sentences) > num_sentences:
            combined_summary = '. '.join(sentences[:num_sentences]) + '.'

        summary_words = combined_summary.split()
        if len(summary_words) > word_limit:
            combined_summary = ' '.join(summary_words[:word_limit]) + '...'

        return combined_summary
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return ''
    
def summarize_content_updated(content, num_sentences, word_limit):
    try:
        # Use the tokenizer and summarizer initialized globally
        global tokenizer, summarizer

        max_input_tokens = 1024  # Max tokens for BART model
        max_summary_tokens = int(word_limit / 0.75)  # Approximate tokens in summary

        # Tokenize content to get the number of tokens
        input_tokens = tokenizer.encode(content, return_tensors='pt', truncation=False)
        input_length = input_tokens.shape[1]

        # Handle content longer than max_input_tokens
        if input_length > max_input_tokens:
            # Chunking logic
            chunks = []
            total_tokens = input_length
            start = 0

            while start < total_tokens:
                end = min(start + max_input_tokens, total_tokens)
                chunk_tokens = input_tokens[:, start:end]
                chunk = tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
                chunks.append(chunk)
                start += max_input_tokens

            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                # Tokenize chunk to get its length
                chunk_tokens = tokenizer.encode(chunk, return_tensors='pt', truncation=False)
                chunk_length = chunk_tokens.shape[1]

                # Adjust max_new_tokens and min_new_tokens for the chunk
                chunk_max_new_tokens = min(max_summary_tokens, chunk_length)
                chunk_min_new_tokens = max(5, int(chunk_max_new_tokens * 0.3))

                if chunk_length <= 11:
                    # If the chunk is too short, skip summarization
                    summaries.append(chunk.strip())
                    continue

                # Summarize the chunk
                summary = summarizer(
                    chunk,
                    max_new_tokens=chunk_max_new_tokens,
                    min_new_tokens=chunk_min_new_tokens,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])

            # Combine summaries
            combined_summary = ' '.join(summaries)
        else:
            # Adjust max_new_tokens and min_new_tokens based on input_length
            max_new_tokens = min(max_summary_tokens, input_length)
            min_new_tokens = max(5, int(max_new_tokens * 0.3))

            if input_length <= 11:
                # If the input is too short, return the original content
                return content.strip()

            # Summarize the content
            summary = summarizer(
                content,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False
            )
            combined_summary = summary[0]['summary_text']

        # Post-process to enforce number of sentences and word limit
        sentences = combined_summary.strip().split('. ')
        if len(sentences) > num_sentences:
            combined_summary = '. '.join(sentences[:num_sentences])
            if not combined_summary.endswith('.'):
                combined_summary += '.'

        summary_words = combined_summary.split()
        if len(summary_words) > word_limit:
            combined_summary = ' '.join(summary_words[:word_limit]) + '...'

        return combined_summary.strip()
    except Exception as e:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error summarizing content: {e}")
        return ''
    

# Define function using google gemini 1.5 flash model to summarize the content
def summarize_content_google(content):
    try:
        # Initialize Google Generative AI API
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        # Use the google gemini 1.5 flash model to summarize the content
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(["Summarize the following text into 6 sentences with a word limit of 200", content])

        summary = clean_text(response.text)
        return summary
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return ''
    
# Define the function to summarize the content using the Groq API
def summarize_content_groq(content):
    try:
        # Use the Hugging Face Hub API to summarize the content
        client = Groq(api_key=os.environ['GROQ_API_KEY'])
        response = client.chat.completions.create(
            #
            # Required parameters
            #
            messages=[
                # Set an optional system message. This sets the behavior of the
                # assistant and can be used to provide specific instructions for
                # how it should behave throughout the conversation.
                {
                    "role": "system",
                    "content": "You are a text summarizer who summarizes long text into a shorter version."
                },
                # Set a user message for the assistant to respond to.
                {
                    "role": "user",
                    "content": f"Summarize this text into 6 sentences with a word limit of 200 words, {content}",
                }
            ],

            # The language model which will generate the completion.
            model="llama3-8b-8192",

            #
            # Optional parameters
            #

            # Controls randomness: lowering results in less random completions.
            # As the temperature approaches zero, the model will become deterministic
            # and repetitive.
            temperature=0.7,

            # Controls diversity via nucleus sampling: 0.5 means half of all
            # likelihood-weighted options are considered.
            top_p=0.85, 

            # A stop sequence is a predefined or user-specified text string that
            # signals an AI to stop generating content, ensuring its responses
            # remain focused and concise. Examples include punctuation marks and
            # markers like "[end]".
            stop=None,

            # If set, partial message deltas will be sent.
            stream=False,
        )

        summ = clean_text(response.choices[0].message.content.strip())
        summ = summ.split(':')[-1].strip()
        return summ
        
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return ''   

# Get the image for the article
def get_article_img(url):
    try:
        driver = webdriver.Chrome() # Start the Chrome browser

        driver.get(url) # Open the URL

        close_overlay(driver) # Close overlays

        close_overlay_new(driver, labels=['Close', 'Dismiss', 'Cancel', 'No Thanks']) # Close overlays with different labels

        buttons = ['Accept', 'Allow',  'Close', 'Cancel',
                   'I Accept','Not Now','Continue to Site','Ok']

        
        for text in buttons:
            click_button_by_text(text,driver) # Accept cookies and close pop-ups

        # # Load more content by clicking on the 'Load More' or 'Read More' button
        load_more_content(driver,'Load More')
        load_more_content(driver,'Read More')

       

        wait = WebDriverWait(driver, 10) # Set the wait time for the browser
        # Wait until the first img tag is loaded or 2 seconds pass
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'img')))

        soup = BeautifulSoup(driver.page_source, 'html.parser') # Parse the HTML content of the page

        driver.quit() # Close the browser

        # Find the first image in the article
        image_tags = soup.find_all('img')
        image = []
        for img in image_tags:
            img_d = img.get('src')
            img_class = img.get('class')
            img_alt = img.get('alt')
            if ('.jpg' in img_d or '.png' in img_d) and (img_class or img_alt):
                image.append(img_d)
                # break

        return image[-1]
    except Exception as e:
        # Save the error logs to the error_logs.log file
        logging.error(f"Error getting image for URL: {url} is {e}")
        return ''

################## FROM HERE ONWARDS THE CODE SCRPAPES,SUMMARIZES, DOWNLOADS IMAGES AND SENDS EMAILS ####################

# Initialize the GoogleNews object
gn = GoogleNews(lang='en', country='IN')

# List of companies to search for
companies = ['IDFC First bank','Fair Isaac Corporation', 'AAVAS financiers','IndoStar Home Finance','DBS Bank (India)','Nice Actimize',
             'Toyota Finance (India)','PNB Housing Finance','UGRO capital','Bajaj Allianz Life Insurance',
             'South Indian Bank','CredAble','MPOWER financing','VGZ Medical Insurance','Rabobank (Dutch Bank)','J & E Davy UK',
             'Volkswagen Finance Private Limited (India)','Further Group Insurance Madrid','Coinit Gold Investments','Bajaj Finance','Aditya Birla Finance']


# List to store the articles
results = []  

# Search for articles for each company

for company in companies:
    # Search the top 3 latest articles for the company
    kw =  'Latest news,updates,market trends,stock market news on '+company # Add the company name to the search query
    kw = '&'.join(kw.split()) # Replace spaces with & in the query
    search_results = gn.search(kw, when='1d') # Search for articles from the past day
    if 'entries' in search_results:
        if len(search_results['entries']) >= 3:
            for entry in search_results['entries'][:3]:
                results.append({
                    'Company': company,
                    'Title': entry.title,
                    'Article_Link': entry.link,
                    'Published': datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z'),
                    'Website': entry.source.get('title','')
                    })
        else:
            for entry in search_results['entries']:
                results.append({
                    'Company': company,
                    'Title': entry.title,
                    'Article_Link': entry.link,
                    'Published': datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z'),
                    'Website': entry.source.get('title','')
                    })
    else:
        pass

time.sleep(0.5) # Add a delay to avoid overloading the server

# Scrape the content of the articles and add it to the results
for result in results:
    url = result['Article_Link']
    article_content = get_article_content(url)
    if len(article_content.split(' ')) < 200 or article_content=='':
        # Fall back to the parse function if the content is too short or empty
        article_content = parse(url)
        if len(article_content.split(' ')) < 200 or article_content=='':
            # Fall back to the get_body_text function if the content is too short or empty
            article_content = get_body_text(url)
            if len(article_content.split(' ')) < 200 or article_content=='':
                # Fall back to the base_scraper function if the content is too short or empty
                article_content = base_scraper(url)
    
    article_content = clean_text(article_content)
    result['Content'] = article_content
    time.sleep(0.5) # Add a delay to avoid overloading the server

# Pause for a few seconds to avoid overloading the server
time.sleep(1)

# Summarize the content of the articles
for result in results:
    content = result['Content']
    num_sentences, word_limit = get_summarization_settings()
    summ = ''
    # Skip summarization if content is too short or empty
    if len(content.split()) >= 200 or content != '':
        # Summarize the content using BART
        summ = summarize_content(content, num_sentences, word_limit)
        # If summarization fails, fall back to the updated function
        if summ == '' or len(summ.split()) < 50:
            summ = summarize_content_updated(content, num_sentences, word_limit)
            # If summarization fails, fall back to the groq function
            if summ == '' or len(summ.split()) < 50:
                summ = summarize_content_groq(content)
                # If summarization fails, fall back to the google function
                if summ == '' or len(summ.split()) < 50:
                    summ = summarize_content_google(content)


        summ =clean_text(summ)
        result['Summary'] = summ
    else:
        result['Summary'] = summ

    # Give a time delay to avoid overloading the server
    time.sleep(random.uniform(2, 3)) # Random delay between 2 and 3 seconds

# Pause for a few seconds to avoid overloading the server
time.sleep(1) 

# Get the image for each article
for result in results:
    url = result['Article_Link']
    img = get_article_img(url)
    result['Image_URL'] = img
    time.sleep(0.5) # Add a delay to avoid overloading the server

# Save all the results to a dictionary and a pickle file
results_dict = {i:results[i] for i in range(len(results)) if results[i]['Summary']!= '' or len(results[i]['Summary'].split()) > 50}

# Save the results to a pickle file with the day's date tag
today = datetime.today().strftime('%Y-%m-%d')
file_name = f'daily_news_{today}.pkl'
pd.to_pickle(results_dict, file_name)

# Log the successful completion of the script and save it a log file
logging.basicConfig(filename='daily_news.log', level=logging.INFO)
logging.info(f"Daily news scraping completed successfully. Results saved to {file_name}")

# # Define the function to download the image
# def download_image(url):
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         return response.content
#     except Exception as e:
#         # Save the error logs to the error_logs.log file
#         logging.error(f"Error downloading image from URL: {url} is {e}")
#         return None


# # Define the function to resize the image
# def resize_image(image_data, max_width=800):
#     try:
#         image = Image.open(io.BytesIO(image_data))
#         if image.width > max_width:
#             ratio = max_width / float(image.width)
#             new_height = int((float(image.height) * float(ratio)))
#             image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
#             img_byte_arr = io.BytesIO()
#             image.save(img_byte_arr, format='JPEG', quality=85)
#             return img_byte_arr.getvalue()
#         else:
#             # Even if resizing is not needed, re-encode the image to ensure consistency
#             img_byte_arr = io.BytesIO()
#             image.save(img_byte_arr, format='JPEG', quality=85)
#             return img_byte_arr.getvalue()
#     except UnidentifiedImageError as e:
#         logging.error(f"UnidentifiedImageError: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error resizing image: {e}")
#         return None

# PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/150'

# def download_image_with_fallback(url):
#     image_data = download_image(url)
#     if image_data:
#         try:
#             Image.open(io.BytesIO(image_data))  # Test if it's a valid image
#             return image_data
#         except UnidentifiedImageError:
#             pass
#     # If image is invalid, download placeholder image
#     logging.info(f"Using placeholder image for URL: {url}")
#     return download_image(PLACEHOLDER_IMAGE_URL)


# Define the function to save the image locally
# def save_image_locally(image_data, image_url):
#     image_name = os.path.basename(image_url)
#     image_path = os.path.join('temp_images', image_name)
#     os.makedirs('temp_images', exist_ok=True)
#     with open(image_path, 'wb') as f:
#         f.write(image_data)
#     return image_path



# Now send the email to a list of recipients with the news articles

# def send_email(articles, sender_email, sender_name, smtp_server, smtp_port, smtp_username, smtp_password,recipients):
#     try:
#         # Download the logo
#         logo_url = 'https://dhurin.in/wp-content/uploads/2022/06/logo-dhurin.png'
#         logo_data = download_image(logo_url)

#         # Start building the HTML content
#         html_content = ""

#         # Add the logo at the top
#         logo_html = f"<img src='cid:logo_image' style='width:100px; display: block; margin-left: auto; margin-right: auto;' />"
#         html_content += logo_html

#         # Add the title
#         # Add title date
#         html_content += f"<h1 style='text-align: center;'>Daily News Feed,{datetime.today().strftime('%a,%b-%y')}</h1>"
#         html_content += "<hr>"

#         # List to hold images to attach
#         attached_images = []

#         # Build HTML content for each article
#         for idx, article in enumerate(articles):
#             article_html = "<div class='article-box'>"
#             article_html += "<table style='width:100%;'><tr>"

#             # Image column
#             article_html += "<td style='width:10%;'>"
#             image_data = None
#             if article.get('Image_URL'):
#                 image_data = download_image_with_fallback(article['Image_URL'])
#             if image_data:
#                 try:
#                     image_data = resize_image(image_data)
#                     if image_data:
#                         image_cid = f"image_{idx}"
#                         attached_images.append({'cid': image_cid, 'data': image_data})
#                         article_html += f"<img src='cid:{image_cid}' style='width:150px; height:80px;' />"
#                     else:
#                         article_html += "Image not found."
#                 except Exception as e:
#                     logging.error(f"Error processing image for article {idx}: {e}")
#                     article_html += "Image not found."
#             else:
#                 article_html += "Image not found."
#             article_html += "</td>"

#             # Text column
#             article_html += "<td style='vertical-align:top;'>"
#             # Title
#             title = f"<a href='{article['Article_Link']}' style='font-size:25px;'>{article['Title']}</a>"
#             article_html += title
#             # Published date
#             p_date = article['Published'].strftime('%a, %d %b %Y')
#             pub_date = f"<p style='font-size:12px;font-weight:bold;'>Published Date: {p_date}</p>"
#             article_html += pub_date
#             # Summary
#             summ = f"<p style='font-size:16px;font-family:Times New Roman;'>{article['Summary']}</p>"
#             article_html += summ
#             article_html += "</td>"

#             article_html += "</tr></table>"
#             article_html += "</div>"

#             html_content += article_html
#             html_content += "<hr>"

#         # Add the logo at the bottom 
#         html_content += logo_html

#         # CSS styles
#         css_styles = """
#             body {
#                 background-color: white;
#             }
#             .article-box {
#                 border: 5px solid pink;
#                 padding: 10px;
#                 margin-bottom: 5px;
#                 box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
#                 border-radius: 10px; /* Adds rounded corners */
#             }
#             """
        
#         # Combine CSS and HTML content
#         full_html = f"""
#         <html>
#         <head>
#             <style type="text/css">
#                 {css_styles}
#             </style>
#         </head>
#         <body>
#             {html_content}
#         </body>
#         </html>
#         """

#         # Inline CSS
#         inlined_html = transform(full_html)

#         # Create the email message
#         msgRoot = MIMEMultipart('related')
#         msgRoot['Subject'] = "Daily News Feed"
#         msgRoot['From'] = f"{sender_name} <{sender_email}>"
#         msgRoot['To'] = ';'.join(recipients.split(','))

#         all_recipients =  [email.strip() for email in recipients.split(',') if email.strip()]

#         # Encapsulate the HTML content in an 'alternative' part
#         msgAlternative = MIMEMultipart('alternative')
#         msgRoot.attach(msgAlternative)

#         # Attach the HTML content
#         msgText = MIMEText(inlined_html, 'html')
#         msgAlternative.attach(msgText)

#         # Attach the logo image
#         logo_img = MIMEImage(logo_data)
#         logo_img.add_header('Content-ID', '<logo_image>')
#         msgRoot.attach(logo_img)

#         # Attach article images
#         for img in attached_images:
#             image = MIMEImage(img['data'])
#             image.add_header('Content-ID', f"<{img['cid']}>")
#             msgRoot.attach(image)

#         # SMTP server configuration
#         smtp_server = os.environ['SMTP_SERVER']
#         smtp_port = int(os.environ['SMTP_PORT'])
#         smtp_username = os.environ['SMTP_USERNAME']
#         smtp_password = os.environ['SMTP_PASSWORD']

#         # Send the email
#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(smtp_username, smtp_password)
#             server.sendmail(sender_email, all_recipients, msgRoot.as_string())

#         logging.info("Email sent successfully")
#     except Exception as e:
#         # Save the error logs to the error_logs.log file
#         logging.error(f"Error sending email: {e}")
#         pass

# Send the email with the news articles
# articles = []

# for art in results_dict:
#     articles.append(results_dict[art])

# send_email(results, os.environ['SENDER_EMAIL'], os.environ['SENDER_NAME'], os.environ['SMTP_SERVER'],
#            os.environ['SMTP_PORT'], os.environ['SMTP_USERNAME'], os.environ['SMTP_PASSWORD'],
#            os.environ['RECIPIENTS'])

# Delete all the temporary files created and other files ,dicionaries,directories and variables
del results # Delete the results list
del results_dict # Delete the results dictionary
del gn # Delete the GoogleNews object
del companies # Delete the companies list
del summarizer # Delete the summarizer object
del tokenizer # Delete the tokenizer object




## Or we can schedule the script to run in a Heroku environment using the Heroku Scheduler add-on
# Add the Heroku Scheduler add-on to your Heroku app and schedule the script to run daily at a specific time.

# Now we will connect to our Heroku server and run the daily_news_app.py file
# First, we need to install the Heroku CLI and log in to our Heroku account
# After logging in, we can run the following command to run the daily_news_app.py file on the Heroku server
# heroku run python daily_news_app.py
# This will execute the script on the Heroku server and send the daily news summary email to the recipients.







    

