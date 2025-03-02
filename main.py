from newspaper import Article
from newsapi import NewsApiClient
#import facebookdata
from openai import OpenAI
import requests
from transformers import pipeline
#import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import re
import sys

#BAD WORD = LOADING, ERROR, ADVERTISEMENT, Advertisement
import random
import os

from news_sentiment_anal import text_summary_small, text_summary_long 
from wordpress_upload import postToFacebook, wordpress
import sys
from dotenv import load_dotenv

#random.seed(8)

def main():

    load_dotenv('.env.news')

    
    Topics = ["Hungary", "Politics", "Economics", "Orbán", "Science", "Food", "Environment", "Fishing", "Cooking", "Film", "Travelling", "Finance"]
    randomtopic = random.choice(Topics)
    print(randomtopic)
    newsapi = os.environ["NEWSAPI"]
    news = NewsApiClient(api_key=newsapi)
    client = OpenAI(api_key=os.environ["DEEPSEEKAPI"], base_url="https://api.deepseek.com")


    #Sample url
    # url = "https://www.euronews.com/green/2024/05/01/weeds-for-the-win-how-to-turn-your-garden-into-a-haven-for-insects"
    # article = Article(url)
    # article.download()
    # article.parse()
    # text = article.text
    # print(text)
    # exit()
    # clean_text = text.replace('\n', ' ')


    counter = 0
    AppropriateNews = False
    while AppropriateNews == False:

        all_articles = news.get_everything(q=randomtopic,
                                      language='en',
                                      sort_by='relevancy')

        RandomURL = random.randint(0, len(all_articles['articles']))

        url = all_articles['articles'][RandomURL]['url']
        article = Article(url)

        try:
            article.download()
            article.parse()
        except:
            print("Nem sikerült")
            counter += 1
            if counter > 5:
                randomtopic = random.choice(Topics)
            continue
        text = article.text
        Title = article.title
        clean_text = text.replace('\n', ' ')
        if len(clean_text) > 3000:
            AppropriateNews = True

    print(clean_text)
    print('---------------------')

    main_article_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system", "content":"You are an article writer, who creates long comprehensize SEO optimized summarizes from articles in 2-4 paragpraphs and translates from english to hungarian. No need for english version."},
            {"role": "user", "content": f"{clean_text}"}
        ]
    )

    main_article = main_article_response.choices[0].message.content
    #main_article = main_article.replace("\n", "<br>")

    main_lead_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system", "content":"You are a facebook lead writer, who writes 3 or 4 sentences long SEO optimized leads, from an article and translates them from english to hungarian. No need for english version."},
            {"role": "user", "content": f"{clean_text}"}
        ]
    )

    main_lead = main_lead_response.choices[0].message.content

    title_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system", "content":"Give an eye catching title without apostrophe for the article in hungarian. Only Title, no other."},
            {"role": "user", "content": f"{clean_text}"}
        ]
    )

    keyword_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system", "content":"Give me one main keywords without apostrophe, for the article. No more."},
            {"role": "user", "content": f"{clean_text}"}
        ]
    )
    keyword = keyword_response.choices[0].message.content

    image_name, image_url = image_search(keyword)

    
    title = title_response.choices[0].message.content
    print(title)
    print("-----------------")
    print(main_article)
    print("-----------------")
    print(main_lead)

    html_content = """
    <div class="wp-block-columns is-layout-flex wp-container-core-columns-is-layout-1 wp-block-columns-is-layout-flex">
        <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">
            <figure class="wp-block-image size-large"><img fetchpriority="high" fetchpriority="high" decoding="async" width="1024" height="576" src="https://tipplee.hu/wp-content/uploads/2024/07/torpeharcsa-1024x576.jpg" alt="törpeharcsa pucolás" class="wp-image-1734" srcset="https://tipplee.hu/wp-content/uploads/2024/07/torpeharcsa-1024x576.jpg 1024w, https://tipplee.hu/wp-content/uploads/2024/07/torpeharcsa-300x169.jpg 300w, https://tipplee.hu/wp-content/uploads/2024/07/torpeharcsa-768x432.jpg 768w, https://tipplee.hu/wp-content/uploads/2024/07/torpeharcsa.jpg 1200w" sizes="(max-width: 1024px) 100vw, 1024px" /></figure>
            <p><a href="https://tipplee.hu/torpeharcsa-pucolas-4-lepeses-hasznos-utmutato-1714/">Törpeharcsa pucolás: 4 lépéses hasznos útmutató</a></p>
        </div>

        <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">
            <figure class="wp-block-image size-large"><img decoding="async" width="1024" height="576" src="https://tipplee.hu/wp-content/uploads/2024/11/rekord-vorosszarnyu-1024x576.jpg" alt="rekord vörösszárnyú" class="wp-image-3753" srcset="https://tipplee.hu/wp-content/uploads/2024/11/rekord-vorosszarnyu-1024x576.jpg 1024w, https://tipplee.hu/wp-content/uploads/2024/11/rekord-vorosszarnyu-300x169.jpg 300w, https://tipplee.hu/wp-content/uploads/2024/11/rekord-vorosszarnyu-768x432.jpg 768w, https://tipplee.hu/wp-content/uploads/2024/11/rekord-vorosszarnyu.jpg 1200w" sizes="(max-width: 1024px) 100vw, 1024px" /></figure>
            <p><a href="https://tipplee.hu/lenyugozo-rekord-vorosszarnyu-keszeg-1-kg-nal-3750/">Lenyűgöző rekord vörösszárnyú keszeg: 1 kg-nál mennyivel nagyobb?</a></p>
        </div>

        <div class="wp-block-column is-layout-flow wp-block-column-is-layout-flow">
            <figure class="wp-block-image size-large"><img decoding="async" width="1024" height="576" src="https://tipplee.hu/wp-content/uploads/2025/01/Csuka_allomany-1024x576.jpg" alt="Csukaállomány" class="wp-image-4332" srcset="https://tipplee.hu/wp-content/uploads/2025/01/Csuka_allomany-1024x576.jpg 1024w, https://tipplee.hu/wp-content/uploads/2025/01/Csuka_allomany-300x169.jpg 300w, https://tipplee.hu/wp-content/uploads/2025/01/Csuka_allomany-768x432.jpg 768w, https://tipplee.hu/wp-content/uploads/2025/01/Csuka_allomany.jpg 1200w" sizes="(max-width: 1024px) 100vw, 1024px" /></figure>
            <p><a href="https://tipplee.hu/a-csuka-allomany-csokkenese-3-nyomaszto-magyarazat-4331/">A csuka állomány csökkenése – 3 nyomasztó magyarázat</a></p>
        </div>
    </div>
    """

    main_article = "".join(f"<p>{para.strip()}</p>" for para in main_article.split("\n") if para.strip())


    main_article = f"{main_article}<br><br>{html_content}"

    wp_url = wordpress(title = title, body=main_article, cikk_url=url, image_to_upload=image_name, version="V1", excerpt=main_lead, image_url=image_url)
    
    os.remove(image_name)

    sys.exit()
    text = text + f"\n Source:{wp_url}"
    for i in keywords:
        text = text + f" #{i}, "
    
    #postToFacebook(text)



def image_search(search_term):
    access_key = os.environ["UNSPALSHAPI"]
    url = f"https://api.unsplash.com/search/photos?query={search_term}&client_id={access_key}"

    notsiker = True
    counter = 0
    while notsiker:
        try:
            response = requests.get(url).json()
            notsiker = False
        except:
            print("Nem találtam képet")
            counter += 1
            print(counter)

    if "results" in response:
        image_url = response["results"][0]["urls"]["raw"] + "&w=1200&h=675&fit=crop"
        image_name = f"{search_term}_1200x675.webp"
        img_data = requests.get(image_url).content
        with open(image_name, "wb") as img_file:
            img_file.write(img_data)
        sourceurl = response["results"][0]['links']['download']
    return image_name,  sourceurl




if __name__ == "__main__":
    main()