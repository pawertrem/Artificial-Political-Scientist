# Artificial-Political-Scientist
A code for building a GPT-enabled RETRO (Retrieval-Enhanced Transformer) bot, which would provide political analysts with informational support. 
A bot's architecture can be considered as two-fold one:

Semantic Search - transformer model encodes a question and finds for it a vector with the highest dot product among preliminary encoded text fragments
Articulation of answer with Text-Davinci-003 on the basis of text fragments with the highest dot products
Data is presented by fragmented content of Russian political science journal "Citizen. Elections. Authority" (https://www.rcoit.ru/lib/gvv/)
