# projet 7, Speech to text (+ translation)
# transformer model, local, no API
# pytorch, dataset fr

# projet original : https://www.kaggle.com/code/stpeteishii/french-audio-wav2vec2-translation
# pbblmt inspiré (?) par : https://www.kaggle.com/code/dikshabhati2002/speech-to-text-with-hugging-face/notebook

# Il y a encore 2 ans, le modèle wav2vec2 de facebook obtenait les meilleures performances en termes de transcription speech to text.
# Les espaces d'embedding contextualisés, similaires à ceux qu'on utilise en NLP (word2vec, doc2vec, glove, etc...)
# semblaient être l'approche la plus prometteuse pour construire un "traducteur universel".

# Il y a un an environ, l'architecture des transformers encoder-decoder, similaire à celle des LLMs, a révolutionné le domaine.


# Ce projet comporte 2 notebooks + une appli


	# 1 Présentation du dataset de test, EDA rapide

? note sur preprocessing audio classique ? mel-freq, spectrogramme (+ FFT) + CNN (par exemple)

# evaluation objective et quantifiée de la transcription par le modèle original
# metrique : WER(Word error rate) 

# evaluation de la traduction (bonus)
# plus difficile à quantifier, observation sur qq exemples


	# 2 Nouvelle approche : whisper

# transcription
# traduction


	# 3 Appli locale streamlit

# Il existe de nombreuses API que l'on peut aujourd'hui facilement requêter, par exemple avec langchain, ou directement sur vortex.
# Cela peut permettre une économie en ressources.

# Pour ce projet, on peut imaginer que pour des raisons de sécurité ou de confidentialité on souhaite plutôt utiliser un modèle local.
# Cette solution présente aussi l'avantage de fonctionner hors-connexion.

