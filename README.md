# PFE_Project
PFE sur le Fine tuning de modèles d'embeddings : Comment mettre en place un processus itératif de fine-tuning d’un modèle d’embedding en intégrant progressivement du jargon métier, tout en automatisant les étapes clés de ce pipeline ?

Benchmark_all_models_before_ft_baconnier.ipynb:
Ce code évalue les performances de plusieurs modèles d'embedding de base (Solon, BGE-M3, E5, etc.) sur le dataset Bercy (Baconnier) avant tout entraînement. Il génère un classement comparatif des modèles selon des métriques de recherche (Recall, MRR, nDCG).

Benchmark_all_models_after_ft_baconnier.ipynb:
Ce notebook compare les performances des modèles d'embedding après leur fine-tuning sur le dataset Bercy (Baconnier). Il permet de mesurer concrètement l'apport des différentes configurations d'entraînement par rapport aux versions de base.

Dataset_per_and_all_domains_verif_and_split_train_test.ipynb:
Ce script est dédié à la préparation et à l'exploration des données (termes, acronymes, domaines). Il vérifie l'intégrité et la répartition du dataset, puis s'occupe de la séparation des données en jeux d'entraînement et de test.

Benchmark_all_models_after_ft_bercy_set (1).ipynb:
Ce code évalue les modèles (de base et fine-tunés) selon deux critères métiers spécifiques : une évaluation « Concept » (tolérante sur le contexte) et une évaluation « Def-only » (stricte sur la recherche de définition exacte). Il permet de juger la capacité des modèles à se comporter de manière pertinente pour un cas d'usage de type RAG.

Fine_tuning_Solon_large (1).ipynb:
Ce notebook réalise le fine-tuning du modèle d'embedding français Solon-embeddings-large pour la recherche de termes administratifs. Il implémente des optimisations parameter-efficient (LoRA sur les projections K et Q) combinées à une stratégie de loss avec in-batch negatives.

Fine_tuning_e5_large-instruct (2).ipynb:
Ce code configure et lance le fine-tuning du modèle multilingual-e5-large-instruct sur le jeu de données administratif. Il adapte le modèle à la tâche spécifique d'extraction d'acronymes à l'aide de préfixes d'instruction (Prompt/Query) et d'un entraînement LoRA.

app_ministere_rag_v_5.py:
Il s'agit d'une version évoluée de l'application RAG qui ajoute une couche de pré-traitement robuste aux requêtes utilisateurs (correction orthographique, détection de langue et traduction). Le script enrichit également l'interface d'administration avec des détails de débogage sur la transformation des questions et les scores de pertinence.

Evaluation_RAGAS_Solon_Baseline_vs_Finetuned.ipynb:
Ce notebook utilise la librairie RAGAS pour évaluer objectivement la qualité des réponses générées par le modèle Solon (avant et après fine-tuning). Il calcule des métriques clés comme la précision du contexte (Context Precision) et le rappel (Context Recall) pour valider l'amélioration du modèle.

Evaluation_RAGAS_E5_Baseline_vs_Finetuned.ipynb:
Sur le même principe, ce code compare les performances du modèle E5-large-instruct dans ses versions baseline et fine-tunée. Il permet de quantifier les gains de performance apportés par l'entraînement spécifique sur le corpus métier à l'aide des métriques RAGAS.


test_et_comparaisons_des_modèles (1).ipynb:
Ce script réalise un benchmark comparatif technique entre plusieurs modèles d'embedding (Solon, E5, GTE, Marsilia, etc.) sur des tâches de similarité pure. Il analyse leur capacité à rapprocher les paires positives (Retrieval) et à distinguer les concepts proches via des tests sur des triplets (Séparation).


Benchmark_all_models_before_ft_bercy_set.ipynb:
Ce notebook établit les scores de référence (baseline) de tous les modèles sur le jeu de données complet de Bercy avant tout réentraînement. Il applique les protocoles d'évaluation « Concept » (regroupement sémantique global) et « Def-only » (recherche de définition stricte) pour fixer le point de départ des comparaisons.


