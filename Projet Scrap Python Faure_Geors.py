#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Methodes du projet
get_ipython().system('pip install textblob')
get_ipython().system('pip install textblob_fr    ')
from requests import get
import bs4 as bs
import urllib.request
import nltk
from bs4 import BeautifulSoup
import string
import numpy as np
import matplotlib
from matplotlib import pyplot as pyp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import RegexpTokenizer
from heapq import nlargest
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer    
nltk.download("stopwords")
nltk.download('punkt')


sw = stopwords.words('french') 
sw.append('a')
sw.append("d'")
sw.append("n'")
sw.append("l'")#Ces mots ne sont pas compté alors qu'ils sont récurrents, ils gênent l'attribution de poids aux phrases des articles
punctuation = string.punctuation + '’«»—.\n'#Ponctuation française trouvé dans les articles de CNEWS
main_url = 'https://www.cnews.fr/'
test_url = '/culture/2021-01-05/pas-de-date-de-reprise-pour-le-secteur-culturel-annonce-roselyne-bachelot-1033263'

def Check_Web_Connection(actual_url):
    response = get(actual_url)
    return response

def Get_Web_Soup(response):
    html_soup = BeautifulSoup(response.text,"html.parser")#Récupère l'ensemble du site web 
    return html_soup

def Get_URL_main_articles(soup):#Récupération des articles mis en avant par le site
    body = soup.find(id='main-content')
    temporary_url_array = []
    main_art = body.find_all('div',class_="dm-block dm-block-bloc_1_news")
    for elem in main_art:
        url = elem.find('a',href=True)
        temporary_url_array.append(url['href'])
    url_array = [temporary_url_array[0],temporary_url_array[1],temporary_url_array[2],temporary_url_array[3]]
    #Sur CNews, les derniers gros titres correspondent souvent à de la culture et non à des informations sur l'actualité
    return(url_array)

def Get_URL_sub_articles(soup):#Récupération des autres articles
    body = soup.find(id='main-content')
    temporary_url_array = []
    url_array = []
    main_art = body.find_all('div',class_="dm-block dm-block-bloc_3_news")
    for elem in main_art:
        url = elem.find_all('a',href=True)
        for i in url:
            temporary_url_array.append(i['href'])
    for i in range(15): #Ce nombre d'URL peut être modifié en fonction du nombre d'article qu'on veut résumer
        url_array.append(temporary_url_array[i])
    return(url_array)

def Get_text_from_article(actual):#Récupère l'ensemble des morceaux de texte contenu dans l'article, contient également les textes assosiés aux tweets cités ainsi que les URL que je n'ai pas réussi à retirer 
    actual_url= main_url + actual[1:]
    text = []
    response = Check_Web_Connection(actual_url)
    soup = Get_Web_Soup(response)#On récupère l'ensemble du site avant d'y appliquer des recherches via soup.find
    body= soup.find(id="wrapper-publicite")
    titre = body.find("h1",class_="article-title")
    if titre != None: #Il y a quelques problèmes avec l'uniformité des pages du site web, le titre est récupéré sur certains articles mais pas sur tous
        text.append(titre.text + ".")
        text.append(" ")
    intro = body.find("p", class_="dm_article-chapeau")
    text.append(intro.text)
    text.append(" ")
    content_1 = body.find_all("p",class_="dm_article-paragraph")
    content_2 = body.find_all('p',class_= None)#Certains morceaux de texte ne sont associés à aucune classe
    for elem in content_1:
        text.append(elem.text)
        text.append(' ')
    for elem in content_2:
        text.append(elem.text)
        text.append(' ')
    article = ''.join(text)
    return article

def Print_all_articles():#Affiche tout les articles du site
    #ON commence par afficher les grands articles du site
    response = Check_Web_Connection(main_url)
    soup = Get_Web_Soup(response)
    url_prince = Get_URL_main_articles(soup)
    url_sub = Get_URL_sub_articles(soup)
    print("Voici les articles correspondant aux gros titres de Cnews\n")
    for elem in url_prince:  
        print(Get_text_from_article(elem))
        print("\n"*3)
    print("Voici tout les autres articles proposé par Cnews")
    for elem in url_sub:  
        print(Get_text_from_article(elem))
        print("\n"*3)
        
        
def Get_num(article):#Obtiens un dictionnaire indiquant le nombre d'ittération de chaque mot de l'article
    num = {}
    tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''') #J'utilise RegexpTokenizer pour les mots car en utilisant word_tokenize, j'avais certains problèmes causés par la langue française / les textes dans les tweets
    tok_art = tokenizer.tokenize(article)
    for word in tok_art:
        if word.lower() not in sw:
            if word.lower() not in punctuation: 
                if word in num.keys():
                    num[word] = num[word] + 1 #On compte le nombre de fois qu'un mot apparait dans le texte 
                else:
                    num[word] = 1
    return num


def Get_value_sentence(sentence, num):#Retourne le poids d'une phrase de l'article
    tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    sent_tok = tokenizer.tokenize(sentence)
    count = 0
    for word in sent_tok:#On calcule le poids de chaque mots de la phrase grâce au dictionnaire de fréquence qu'on à déterminé précédemment
        if word.lower() not in sw:
            if word.lower() not in punctuation:
                count = count + num[word]
    count = count / len(sent_tok)#On effectue une moyenne du poids de chaque mots de façon à ne pas avantager les phrases très longues qui peuvent s'avérer ne pas être essentielles
    return count

def Get_weight_sentences(num,article):#Retourne le poids de chaque phrase de l'article
    tok_art = sent_tokenize(article, language='french')#on tokenize l'articles en phrases de façon à pouvoir évaluer le poids de chaque phrase
    weight = {}    
    for sentences in tok_art:
        weight[sentences] = Get_value_sentence(sentences,num)
    max_val = weight.values()
    weight[tok_art[0]] = max(max_val)#Permet d'avoir forcement le titre de l'article dans son résumé ce qui rend la compréhension de l'article plus simple
    return weight  

def Summary_article(article):
    num = Get_num(article)
    weight = Get_weight_sentences(num,article)
    length = int(len(weight)*0.25)#Ce nombre peut être modifié selon la taille de résumé que l'on désire, actuellement, on récupère les 25% des phrases de l'articles qui ont le plus grand poids
    summary_array = nlargest(length,weight,key=weight.get)#Renvoie un tableau contenant un pourcentage des phrases ayant le plus grand poids dans l'article
    summary= ''.join(summary_array)
    return summary

def Sentiment_article(article): #Permet d'obtenir le sentiment dégagé d'un article: 0 étant plutot neutre, 1 positif et -1 négatif
    tok_art = sent_tokenize(article, language='french')
    sentiment = 0
    for sentences in tok_art:
        sentiment = sentiment + TextBlob(sentences,pos_tagger = PatternTagger(), analyzer = PatternAnalyzer()).sentiment[0]
    sentiment = sentiment / len(tok_art)
    return sentiment

def Print_sentiment(sentiment_array): #Permet d'afficher la polarité de tout les articles
    pyp.plot(sentiment_array)
    axes = pyp.gca()
    axes.set_ylim(-1,1)
    pyp.xlabel("Positionnement de l'article dans le fil d'actualité")
    pyp.ylabel("Polarité de l'article (1 = positif, -1 négatif , 0 neutre)")
    pyp.suptitle("Polarité des articles actuels de CNEWS")

def Print_detail_article():
    reponse = Check_Web_Connection(main_url)
    main_soup = Get_Web_Soup(reponse)
    text = Get_text_from_article(test_url)
    print("L'article \n")
    print(text + "\n"*4)
    num = Get_num(text)
    sorted_num = {}
    sorted_keys = sorted(num,key=num.get, reverse = True)
    for elem in sorted_keys:
        sorted_num[elem]=num[elem]
    num_list = sorted_num.items()
    temp_x,temp_y = zip(*num_list)
    x=temp_x[:7]
    y= temp_y[:7]
    pyp.plot(x,y)
    pyp.ylabel("Nombre d'ittération")
    pyp.xlabel("Mots les plus employés dans l'article")
    pyp.suptitle("Répartition des mots dans l'article")
    print("\n"*4)
    print("Le poids de chaque phrase de l'article")
    print(Get_weight_sentences(num,text))
    print("\n"*4)
    print("Le résumé de l'article : " + Summary_article(text))
    print("\n"*4 + "La polarité de l'article (-1 négatif, 0 neutre, 1 positif)")
    print(Sentiment_article(text))
    
def Synthese_sentiment():
    reponse = Check_Web_Connection(main_url)
    main_soup = Get_Web_Soup(reponse)
    url_princ = Get_URL_main_articles(main_soup)
    url_sub = Get_URL_sub_articles(main_soup)
    sent_array = []
    for articles in url_princ:
        text = Get_text_from_article(articles)
        sent_array.append(Sentiment_article(text))
    
    for articles in url_sub:
        text = Get_text_from_article(articles)
        sent_array.append(Sentiment_article(text))
    Print_sentiment(sent_array)
    
def Resume_articles():
    reponse = Check_Web_Connection(main_url)
    main_soup = Get_Web_Soup(reponse)
    url_princ = Get_URL_main_articles(main_soup)
    url_sub = Get_URL_sub_articles(main_soup)
    sent_array = []
    for articles in url_princ:
        text = Get_text_from_article(articles)
        print(Summary_article(text))
        print("\n")
    
    
    for articles in url_sub:
        text = Get_text_from_article(articles)
        print(Summary_article(text))
        print("\n")
    


# In[ ]:


#Impression de tous les articles (en fonction du nombre choisi dans la méthode Get_Url_Sub_articles)
Print_all_articles()


# In[ ]:


#Résumé des articles
Resume_articles()


# In[ ]:


#Graph montrant les sentiments dégagés des articles Cnews
Synthese_sentiment()


# In[ ]:


#Méthode montrant en détail le procédé de synthèse d'un article (le graphique se positionne en bas naturellement)
Print_detail_article()


# In[ ]:




