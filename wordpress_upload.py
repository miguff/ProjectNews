#WP_jelszÃ³ a bothoz a mig-wp-hez: 0ypQ By11 fvpA IaOU 4BZd LNDx

import requests
import json
import random
import base64
from requests.auth import HTTPBasicAuth
import os
#import facebookdata


def wordpress(title, body, cikk_url, version, image_to_upload = None, excerpt=None, image_url = None):
    url = os.environ["TIPPLEURL"]
    base = os.environ["TIPPLEBASE"]

    user = os.environ["USER"]
    password = os.environ["MIGSITEPW"]
    robot_pw = os.environ["ROBOTPW"]
    pw_bot = os.environ["PWBOT"]
    tipplePW = os.environ["TIPPLEPW"]
    creds = user + ':' + tipplePW

    token = base64.b64encode(creds.encode())

    header = {'Authorization':'Basic ' + token.decode('utf-8')}



    if image_to_upload == None:
        post = {
        'title':f'{title}',
        'content':f'<!-- wp:paragraph -->{body}  <br><br>  Ez a cikk a Neural News AI ({version}) verziójával készült. Forrás: {cikk_url}<!-- /wp:paragraph --> ',
        'status':'publish',
        'categories': [267],
        "comment_status":"closed",
        "ping_status": "closed",
        "excerpt": f'{excerpt}'
        }

    else:
        media = {
        'file': open(image_to_upload, 'rb'),
        'caption':'',
        'Description' : ''
    }

        image = requests.post(url + '/media', headers=header, files=media)
        imageURL = str(json.loads(image.content)['source_url'])
        imagenumber = str(json.loads(image.content)['id'])

        post = {
        'title':f'{title}',
        'content':f'<!-- wp:paragraph -->{body} <br><p> Ez a cikk a Neural News AI ({version}) verziójával készült. <br>Forrás: <a href="{cikk_url}" target="_blank" rel="noopener noreferrer">{cikk_url}</a>. <br>A kép forrása: {image_url}</p><!-- /wp:paragraph -->',
        'status':'publish',
        'categories': [267],
        "comment_status":"closed",
        "ping_status": "closed",
        "excerpt": f'{excerpt}',
        "featured_media" : imagenumber
        }

    r = requests.post(url + "/posts", headers=header, json=post)

    if r.status_code == 201:
        print("Post published successfully!")
    else:
        print("Post failed:", r.text)

    string = post["title"]
    string = string.lower()
    string = string.replace(',','').replace(":",'').replace(" ", "-").replace('Ã¡','a').replace('Ã©','e').replace('Ã¶','o').replace('Ã¼','u').replace('Ã³','o').replace('Å','o').replace('Ãº','u').replace('Å±','u').replace('Ã­','i')
    url_final = base + string
    return url_final



def postToFacebook(text):
    facebook = os.environ["facebookKey"]
    pageid = os.environ["pageId"]
    post_url = 'https://graph.facebook.com/{}/feed'.format(pageid)
    payload = {
    'message': text,
    'access_token': facebook
    }
    r = requests.post(post_url, data=payload)
    print(r.text)