import streamlit as st
from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Setup chemin src
# ---------------------------------------------------------------------------
def _ensure_src_on_path():
    root = Path(__file__).resolve()
    for p in [root, *root.parents]:
        if (p / "pyproject.toml").exists():
            project_root = p
            break
    else:
        project_root = Path.cwd()
    for add in (project_root, project_root / "src"):
        s = str(add)
        if s not in sys.path:
            sys.path.insert(0, s)
    return project_root


PROJECT_ROOT = _ensure_src_on_path()

try:
    from src.data_visualization import (
        rating_distribution,
        recipe_mean_rating_distribution,
        top_users_by_activity,
        user_mean_rating_distribution,
        user_count_vs_mean_rating,
        activity_bucket_bar,
        analyze_contributors,
        statistique_descriptive,
        plot_prep_time_distribution,
        plot_ingredient,
        plot_n_steps_distribution,
        analyse_tags,
        plot_tags_distribution,
        plot_nutrition_distribution,
        analyze_ingredients_vectorized,
        minutes_group_negative_reviews_bar,
        spearman_correlation,
        plot_ingredients_vs_negative_score,
        analyze_tags_correlation,
        nutrition_correlation_analysis,
        get_most_negative_user,
        plot_minutes_ningredients_nsteps,
    )
    from src.preprocessing import (
        detect_missing_values,
        detect_duplicates,
        remove_outliers_nutrition,
        clean_review,
        is_negative_sentence,
        binary_sentiment,
    )

    DV_OK = True
    DV_ERR = None
except Exception as e:
    DV_OK = False
    DV_ERR = e

from src.streamlit.app.utils import get_ds, render_viz, _safe_rerun


# ---------------------------------------------------------------------------
# Page Visualisations
# ---------------------------------------------------------------------------
def show_visualizations():
    st.markdown("## Visualisations")

    df_merged = get_ds()["merged"]
    df_inter = get_ds()["interactions"]
    df_clean_recipes = get_ds()["clean_recipes"]

    tabs = st.tabs(
        ["Distribution", "Utilisateurs", "Recettes", "Corrélations", "Nutrition"]
    )

    with tabs[0]:
        render_viz("Distribution brute des notes", rating_distribution, df_inter)

        # imputation
        st.markdown(
            """L’analyse univariée de la variable rating met en évidence une forte concentration des évaluations sur les valeurs hautes de l’échelle (4 et 5).
            Ce biais positif traduit une satisfaction générale élevée sur la plateforme, mais il indique aussi que la variable rating n’est pas symétriquement distribuée."""
        )

        render_viz(
            "Analyse des contributeurs des recettes",
            analyze_contributors,
            df_clean_recipes,
        )

        st.markdown(
            """Parmi tous les contributeurs, plus de la moitié ont posté 1 seule recette. 
    Le contributeur le plus actif a posté à lui seul quasiment 3000 recette. 
    Nous pouvons aussi observer que plus de 50% des recettes sont produits par le top 500 des contributeurs parmi les 27523 contributeurs"""
        )

    with tabs[1]:
        render_viz("Top utilisateurs actifs", top_users_by_activity, df_inter)
        render_viz("Activité (buckets) vs note moyenne", activity_bucket_bar, df_inter)
        render_viz(
            "Nombre d'avis vs moyenne (échantillon)",
            user_count_vs_mean_rating,
            df_inter,
            sample=2500,
        )

        st.markdown(
            """Ce graphique représente la note moyenne des utilisateurs par rapport au nombre d'avis laissé. Nous pouvons faire 2 interprétations remarquable : 
                    premièrement, au cas par cas, la plupart des utilisateurs laisse très peu d'avis (car forte concentration proche de 0). 
                    Deuxièment, nous pouvons voir que les utilisateurs ayant donné le plus d'avis ont une moyenne de note de 4,5 ~ 5, par exemple pour l'utilisateur qui a donné le plus d'avis a une moyenne des notes très proche de 5 ! """
        )

    with tabs[2]:
        render_viz(
            "Temps de préparation (catégories)", plot_prep_time_distribution, df_merged
        )
        st.markdown(
            """La variable minute présente une forte asymétrie à droite. La majorité des recettes ont un temps de préparation assez court mais certaines recettes très longues tirent la moyenne vers le haut.
On peut également observer une dispersion très importante (Variance = 321234.9). 

Nous pouvons prendre en note 2 valeurs particulières :
-   Certaine recette présente un temps de préparation de 0, ce qui peut remettre en question la pertinence de la recette. 
Cependant nous supposons que un temps de 0 correspond à un oubli et que cette valeur ne devrait pas influencer notre étude sur le taux d'insatisfaction des recettes
-   Certaine recette présente un temps de préparation de 43200 minutes, correspondant à un mois de préparation. 
Ce temps de préparation prend probablement en compte certains facteurs tel que la maturation d'un ingrédient. 
Nous décidons aussi de le garder dans notre jeu de donnée d'autant plus que ce temps de préparation long pourrait être facteur d'insatisfaction"""
        )
        st.markdown(
            """Sur ce graphique nous avons catégorisé les temps de préparation : 
- Très rapide : ≤ 15 min
- Rapide      : 16 ~ 30 min
- Moyen       : 31 ~ 60 min
- Long        : 1 ~ 2 heures
- Très long   : 2 ~ 4 heures
- Extrême     : ≥ 4 heures

Nous observons que la plupart des recettes prennent au plus 2 heures de prépartations. 
à noté que les recettes avec 0 min de temps de préparation sont aussi inclus dans la catégorie "très rapide", 
cependant en faisant le calcul ci-dessous nous pouvons voir que ces recettes représentent une très petite partie (0.43%). 
Cela n'influence donc pas la distribution entière, nous pouvons donc affirmer que la plupart des recettes prennent bien au plus 2 heures de préparations"""
        )

        render_viz(
            "Ingrédients + Pareto",
            plot_ingredient,
            df_clean_recipes if df_clean_recipes is not None else df_merged,
        )

        st.markdown(
            """"Les 618 ingrédients les plus utilisés dans les recettes représentent 80% des ingrédients les plus utilisés"""
        )

        render_viz("Nombre d'étapes", plot_n_steps_distribution, df_merged)
        if df_merged is not None and "tags" in df_merged.columns:
            render_viz("Distribution des tags", plot_tags_distribution, df_merged)
            if st.checkbox("Afficher stats tags"):
                try:
                    st.dataframe(analyse_tags(df_merged))
                except Exception as e:
                    st.error(e)
        if df_clean_recipes is not None and st.checkbox("Stats ingrédients vectorisés"):
            st.dataframe(analyze_ingredients_vectorized(df_clean_recipes))

    with tabs[3]:
        if df_merged is not None:
            render_viz(
                "Minutes vs insatisfaction (groupes)",
                minutes_group_negative_reviews_bar,
                df_merged,
            )
            if {"n_ingredients", "negative_reviews"}.issubset(df_merged.columns):
                render_viz(
                    "Ingrédients vs insatisfaction",
                    plot_ingredients_vs_negative_score,
                    df_merged,
                )

                st.markdown(
                    """Les analyses menées sur la durée de préparation et le nombre d’ingrédients montrent qu’il n’existe pas de relation significative entre ces variables et le score d’insatisfaction des utilisateurs.
Autrement dit, une recette longue ou complexe n’est pas forcément plus critiquée qu’une recette simple ou rapide. 
Ces résultats suggèrent que les facteurs techniques — tels que le temps nécessaire ou la complexité de la recette — ne sont pas les principaux déterminants de la satisfaction.
Les utilisateurs semblent plutôt juger les recettes selon des critères qualitatifs (goût, type de plat, attentes personnelles, ou présentation) plutôt que sur la durée ou la difficulté.
Ainsi, le niveau d’insatisfaction ne dépend pas directement de l’effort ou du temps investi, mais davantage de l’expérience gustative perçue ou de la catégorie culinaire.

En conclusion, il devient pertinent de poursuivre l’étude en se concentrant sur le type de recette (tags) afin d’identifier quelles catégories de plats (desserts, plats principaux, boissons, etc.) sont les plus sujettes aux critiques."""
                )

            if "tags" in df_merged.columns and {
                "negative_reviews",
                "total_reviews",
            }.issubset(df_merged.columns):
                render_viz(
                    "Tags vs insatisfaction", analyze_tags_correlation, df_merged
                )

                st.markdown(
                    """Le graphique montre les 10 catégories de recettes (tags) présentant les scores moyens d’insatisfaction les plus élevés.

Les tags comme “equipment”, “meat”, “number-of-servings” ou “main-dish” apparaissent en tête, indiquant que les recettes associées à ces catégories sont plus souvent critiquées par les utilisateurs.

Ces résultats peuvent s’expliquer par plusieurs facteurs :

Les recettes “equipment” ou “meat” nécessitent souvent du matériel spécifique ou une maîtrise technique (cuissons précises, températures, ustensiles adaptés), ce qui augmente le risque d’échec.

Les plats “main-dish” ou “occasion” sont généralement préparés pour des repas importants, donc les attentes des utilisateurs sont plus fortes.


À l’inverse, des tags plus généraux ou rapides comme “60-minutes-or-less” obtiennent des scores d’insatisfaction plus faibles, ce qui reflète des recettes plus simples et accessibles.

Conclusion : Cette analyse confirme que le type de recette (capturé par les tags) joue un rôle déterminant dans la perception des utilisateurs.
Les critiques ne concernent pas tant la durée ou la complexité, mais plutôt le contexte culinaire et les attentes liées à chaque catégorie.

On observe que certains tags très fréquents (comme dietary, main-dish) ont un score d’insatisfaction moyen autour de 2, ce qui indique qu’ils sont à la fois populaires et moyennement critiqués.

En revanche, equipment est peu fréquent mais très critiqué (score proche de 2.8).

Cela signifie que la popularité n’est pas forcément liée à la satisfaction : un tag peut être très commun et pourtant globalement apprécié, ou rare mais très mal noté."""
                )

    with tabs[4]:
        needed = {
            "calories",
            "sugar",
            "protein",
            "sodium",
            "total_fat",
            "carbohydrates",
        }
        if df_merged is None or not needed.issubset(df_merged.columns):
            st.warning("Colonnes nutrition manquantes.")
        else:
            render_viz("Distribution nutrition", plot_nutrition_distribution, df_merged)
            render_viz(
                "Corrélations nutrition ↔ insatisfaction",
                nutrition_correlation_analysis,
                df_merged,
            )

        st.markdown(
            """Les caractéristiques nutritionnelles (calories, sucres, graisses, protéines, etc.) n’ont pas d’influence directe sur la perception ou la satisfaction des utilisateurs.
Que la recette soit grasse, sucrée ou salée, cela n’affecte ni le nombre d’avis négatifs, ni le score global d’insatisfaction."""
        )
