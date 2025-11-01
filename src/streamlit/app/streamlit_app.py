"""Application Streamlit principale (plots + textes explicatifs)."""

import sys, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from src.ensure_data import ensure_data


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


# Initialize logging
try:
    from src.logging_config import get_logger

    logger = get_logger("streamlit")
    logger.info("Starting MangeTaMain Streamlit application")
except Exception as e:
    print(f"Warning: Could not initialize logging: {e}")
    logger = None

# ---------------------------------------------------------------------------
# Imports visualisation
# ---------------------------------------------------------------------------
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


from src.streamlit.app.utils import _ensure_src_on_path, get_ds, render_viz, _safe_rerun
from src.streamlit.app.layouts.page_data_cleaning import show_data_page
from src.streamlit.app.layouts.page_visualisation import show_visualizations
from src.streamlit.app.layouts.page_conclusion import show_conclusion_page


# ---------------------------------------------------------------------------
# Thème
# ---------------------------------------------------------------------------
def set_custom_theme(theme="Clair"):
    if theme == "Sombre":
        colors = {
            "primary": "#ffffff",
            "secondary": "#f0f0f0",
            "background": "#181a1b",
            "text": "#e0e0e0",
            "header_gradient": "linear-gradient(135deg,#444,#111)",
            "sidebar_gradient": "linear-gradient(180deg,#333,#111)",
            "button_text": "#ffffff",
        }
    else:
        colors = {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "background": "#F0F2F6",
            "text": "#2C3E50",
            "header_gradient": "linear-gradient(135deg,#667eea,#764ba2)",
            "sidebar_gradient": "linear-gradient(180deg,#667eea,#764ba2)",
            "button_text": "#ffffff",
        }
    st.markdown(
        f"""
    <style>
    .stApp {{background:{colors['background']}; color:{colors['text']};}}
    .main-header {{background:{colors['header_gradient']}; padding:1.4rem; border-radius:14px;
                   color:{colors['button_text']} !important; text-align:center; margin-bottom:1.0rem;}}
    [data-testid="stSidebar"] {{background:{colors['sidebar_gradient']};}}
    [data-testid="stSidebar"] * {{color:#ffffff !important;}}
    .stButton>button {{
        background:{colors['header_gradient']}; color:{colors['button_text']} !important;
        border:none; border-radius:24px; padding:0.55rem 1.3rem; font-weight:600;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page Accueil
# ---------------------------------------------------------------------------
def show_home_page():
    try:
        # Utilisez les datasets depuis le session_state si disponibles
        if "datasets" in st.session_state:
            ds = st.session_state.datasets
        else:
            ds = get_ds()

        recipes_df = ds["clean_recipes"]
        raw_interactions = ds["raw_interactions"]

        # Vérifiez que les données sont valides
        if recipes_df is None or recipes_df.empty:
            st.error("Les données des recettes ne sont pas disponibles")
            return

        if raw_interactions is None or raw_interactions.empty:
            st.error("Les données d'interactions ne sont pas disponibles")
            return

        st.markdown("## INTRODUCTION")
        st.markdown(
            """Notre équipe composé de Guy, Mohamed, Leonnel, Omar et Osman avons décidé de travailler sur le projet MangeTaMain sur la problématique 
                du **taux d'insatisfaction des recettes** en nous basant sur 2 tables : les recettes et les interactions utilisateurs (notes, avis).
                Voici un aperçu de la table recette :
                    """
        )

        st.dataframe(recipes_df.head(5), use_container_width=True)

        st.markdown("""et voici un aperçu de la table interactions :""")

        st.dataframe(raw_interactions.head(5), use_container_width=True)

        st.markdown(
            """Le travaille se décline en plusieurs étapes :  
                    - **data_cleaning** : comprendre la structure, les types de données, les valeurs manquantes, les valeurs aberrantes, etc.  
                    - **Analyse univariée** : analyse statistique descriptive et visualisations pour comprendre les tendances, les distributions, les corrélations, etc.  
                    - **Analyse bivariée** : exploration des relations entre les variables, identification des facteurs influençant le taux d'insatisfaction.  
                    - **Ouverture** : Conclusion et suggestions pour la poursuite de l'analyse  
                    """
        )

        # Boutons navigation rapides
        st.markdown("### Navigation rapide")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Aller à la page data cleaning"):
                _set_page_by_key("data")
                _safe_rerun()
        with c2:
            if st.button("📈 Aller aux Visualisations"):
                _set_page_by_key("viz")
                _safe_rerun()

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        st.write(f"Type d'erreur : {type(e).__name__}")
        if logger:
            logger.error(f"Error in show_home_page: {str(e)}")

        # Informations de debug
        if hasattr(st, "secrets"):
            st.write("Secrets disponibles ✅")
        else:
            st.write("Secrets non disponibles ❌")


PAGES_ORDER = [
    ("🏠 Accueil", "home", lambda: show_home_page()),
    ("📊 Données cleaning", "data", lambda: show_data_page()),
    ("📈 Visualisations", "viz", lambda: show_visualizations()),
    ("📝 Conclusion", "conclusion", lambda: show_conclusion_page()),
]


def _init_page_state():
    if "current_page_idx" not in st.session_state:
        st.session_state.current_page_idx = 0


def _go_delta(delta: int):
    st.session_state.current_page_idx = (
        st.session_state.current_page_idx + delta
    ) % len(PAGES_ORDER)


def _set_page_by_key(page_key: str):
    for i, (_, key, _) in enumerate(PAGES_ORDER):
        if key == page_key:
            st.session_state.current_page_idx = i
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MangeTaMain", page_icon="🍽️", layout="wide")
    _init_page_state()
    if logger:
        logger.info("Initializing Streamlit application main interface")
    if "data_ready" not in st.session_state:
        try:
            st.write("🔄 Chargement des données en cours...")
            ensure_data()
            st.session_state.data_ready = True
            st.write("✅ Données chargées avec succès")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
            st.write(f"Type d'erreur : {type(e).__name__}")
            return

    if "theme" not in st.session_state:
        st.session_state.theme = "Clair"
    set_custom_theme(st.session_state.theme)

    if logger:
        logger.debug(f"Theme set to: {st.session_state.theme}")

    try:
        ds = get_ds()
        if logger:
            logger.info("Successfully loaded datasets for main interface")
            for key, df in ds.items():
                if df is not None:
                    logger.debug(f"Dataset '{key}': {df.shape}")
                else:
                    logger.warning(f"Dataset '{key}' is None")
    except Exception as e:
        if logger:
            logger.error(f"Error loading datasets in main: {str(e)}")
        st.error("Erreur lors du chargement des données")
        return

    # Barre supérieure avec chevrons
    top_left, top_center, top_right = st.columns([0.7, 5, 0.7])
    with top_left:
        if st.button("◀"):
            if logger:
                logger.debug("User navigated to previous page")
            _go_delta(-1)
            _safe_rerun()
    with top_center:
        label, key, _ = PAGES_ORDER[st.session_state.current_page_idx]
        st.markdown(
            f"<h2 style='text-align:center'>{label}</h2>", unsafe_allow_html=True
        )
    with top_right:
        if st.button("▶"):
            if logger:
                logger.debug("User navigated to next page")
            _go_delta(1)
            _safe_rerun()

    with st.sidebar:
        st.markdown("### Pages")
        selected = st.radio(
            "Aller à",
            [lbl for (lbl, _, _) in PAGES_ORDER],
            index=st.session_state.current_page_idx,
        )
        if PAGES_ORDER[st.session_state.current_page_idx][0] != selected:
            for i, (lbl, key, _) in enumerate(PAGES_ORDER):
                if lbl == selected:
                    st.session_state.current_page_idx = i
                    if logger:
                        logger.info(f"User navigated to page: {lbl} (key: {key})")
                    _safe_rerun()
                    break
        st.selectbox(
            "Thème",
            ["Clair", "Sombre"],
            index=["Clair", "Sombre"].index(st.session_state.theme),
            key="theme_selector",
            on_change=lambda: st.session_state.update(
                theme=st.session_state.theme_selector
            ),
        )
        if not DV_OK:
            st.caption(f"⚠️ Module viz: KO ({DV_ERR})")

    st.markdown(
        """
    <div class="main-header">
      <h1>🍽️ MangeTaMain</h1>
      <p>Analyse des recettes et interactions (plots & explications)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    _, _, render = PAGES_ORDER[st.session_state.current_page_idx]
    render()


if __name__ == "__main__":
    main()
