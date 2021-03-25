## Introduction
Voici le rapport mon projet Python de Web Scraping des journaux. Dans ce rapport, j’aborderai le 
fonctionnement du programme ainsi que les difficultés que j’ai rencontrées. Enfin, je me focaliserai 
sur les pistes d’améliorations de mon programme.
## Mon projet
Mon objectif durant ce projet était de faire du scraping des journaux afin de récupérer ses articles 
pour en faire une synthèse de l’actualité. Pour ce faire, j’ai utilisé BeautifulSoup et request de façon 
à récupérer le corps du site que je voulais traiter : CNews. A l’aide de BeautifulSoup, je récupère 
l’ensemble des url des articles avant d’employer une méthode pour stocker tout le texte qu’ils 
contiennent.
Pour la partie du traitement de texte, j’ai utilisé principalement NLTK pour la tokenization. J’ai ainsi 
récupéré une liste de mots souvent employés dans la langue française et qui n’ont donc pas d’interet 
en ce qui concerne l’analyse du sujet de l’article. A ces mots j’ai rajouté à la main la ponctuation que 
l’on peut trouver dans les articles de CNews ainsi que différentes lettres pouvant être considéré 
comme des stopwords mais qui n’était pas dans le module nltk.
Au départ pour la tokenization j’ai utilisé word_tokenize et send_tokenize. En revanche, 
word_tokenize me posait des problèmes de dictionnaire car il découpait différemment les phrases en fonction de la méthode dans laquelle il était appelé, j’ai donc utilisé un autre module qui 
s’appelle RegexpTokenizer et qui m’a permis de résoudre le problème.
Après la tokenization, je détermine le nombre de fois qu’un mot est employé dans l’article de façon 
à lui attribuer une valeur qui sera ensuite utilisée pour déterminer quelle phrase est importante.
A l’aide de ce dictionnaire de valeur, je calcul la valeur de chaque phrase en fonction des mots 
qu’elles contiennent. Ainsi, j’obtiens un classement des phrases jugées par le programme comme 
étant les plus représentatives de l’article. J’utilise ensuite nlargest de la bibliothèque heapq pour me 
renvoyer un pourcentage défini à l’avance des phrases ayant la plus grande valeur, ce qui me permet 
de construire le résumé de l’article. Vous pourrez regarder le procéssus en détail en utilisant la 
méthode Print_detail_article()
Enfin, j’ai décidé d’implémenter une analyse des sentiments dégagés par les articles de Cnews. Pour 
cela j’ai utilisé le module NLP textblob et sa variante qui m’intéressait : textblob_fr qui est un 
module d’analyse de sentiment en français. Je voulais initialement construire mon propre modèle 
sk_learn, mais je n’ai pas trouvé de dataset correspondant à ce que je recherchais pour l’entrainer et 
je n’avais pas le temps de vous demander.
