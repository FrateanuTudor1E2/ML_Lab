import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import gamma

# Sunt la un fast food si stam la o coada cu 2 oameni in fata noastra.
# Unul dintre ei este servit iar celalalt asteapta
# Timpul de servire S1 si S2 sunt independente, variabile aleatoare cu media de 2 minute, rata de servire este de .5/minut
# Care este prob ca voi astepta mai mult de 5 min?
x=np.linspace(0,100,1000)
k=5
lamda=0.5
y = gamma.pdf(x,k,1/lamda)
plt.title("PDF of Gamma Distribution (k = 5)")
plt.xlabel("T")
plt.ylabel("Probability Density")
plt.plot(x,y,label="lamda = 0.5", color='blue')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right',borderaxespad=1, fontsize=12)
plt.ylim([0, 0.40])
plt.xlim([0, 20])
plt.savefig('gamma_lambda.png')
plt.clf()




