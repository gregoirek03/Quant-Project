{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33d3e3f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "# 1. On fixe la graine aléatoire pour avoir les mêmes résultats\n",
    "np.random.seed(42) # le hasard est le même à chaque lancer du script\n",
    "n_samples = 100 # taille de l'échantillon\n",
    "n_features = 2 # nombre de facteurs prédictifs\n",
    "\n",
    "# 2. Génération d'une matrice X (100 lignes, 2 colonnes/facteurs)\n",
    "X = np.random.randn(n_samples, n_features)\n",
    "\n",
    "# 3. On ajoute une colonne de 1 pour l'intercept (la constante beta_0)\n",
    "X_b = np.c_[np.ones((n_samples, 1)), X] \n",
    "\n",
    "# 4. Les vrais coefficients beta théoriques que notre code devra retrouver\n",
    "true_beta = np.array([[4.5], [3.0], [1.2]])\n",
    "\n",
    "# 5. Génération de la cible Y avec un bruit gaussien (epsilon)\n",
    "Y = X_b.dot(true_beta) + np.random.randn(n_samples, 1)\n",
    "\n",
    "print(\"Données générées avec succès !\")\n",
    "print(\"Forme de X_b :\", X_b.shape)\n",
    "print(\"Forme de Y :\", Y.shape)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
