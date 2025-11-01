import os, pathlib, zipfile, io, sys
import streamlit as st

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def _log(msg: str):
    print(f"[ensure_data] {msg}", file=sys.stderr, flush=True)


def _extract_zip(src):
    if isinstance(src, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(src))
    else:
        zf = zipfile.ZipFile(str(src))
    zf.extractall(DATA_DIR)


def _running_on_cloud():
    # Heuristique: présence du mount /mount/src sur Streamlit Cloud
    return "/mount/src" in str(pathlib.Path(__file__).resolve())


def ensure_data():
    # Mode / URL
    mode = (
        st.secrets.get("APP_MODE") if hasattr(st, "secrets") else None
    ) or os.getenv("APP_MODE", "dev")
    remote_url = (
        st.secrets.get("DATA_REMOTE_URL") if hasattr(st, "secrets") else None
    ) or os.getenv("DATA_REMOTE_URL")
    use_sample = (
        st.secrets.get("USE_SAMPLE") if hasattr(st, "secrets") else None
    ) or os.getenv("USE_SAMPLE", "false").lower() in ("1", "true", "yes")

    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    has_raw = any(RAW_DIR.iterdir())
    has_processed = any(PROCESSED_DIR.iterdir())
    _log(
        f"Mode={mode} url={'SET' if remote_url else 'MISSING'} raw={has_raw} processed={has_processed} cloud={_running_on_cloud()} sample_flag={use_sample}"
    )

    # Si déjà extrait
    if has_raw and has_processed:
        return

    # Sans URL -> erreur
    if not remote_url:
        st.error("DATA_REMOTE_URL non défini (ajoute-le dans .streamlit/secrets.toml).")
        return

    # Sample optionnel (désactivé par défaut)
    sample_path = DATA_DIR / "data_sample.parquet"
    if use_sample and sample_path.exists():
        _log("USE_SAMPLE activé -> utilisation du sample parquet")
        return

    # Téléchargement forcé
    st.info("Téléchargement des données…")
    _log(f"Téléchargement: {remote_url}")
    try:
        if "drive.google.com" in remote_url:
            try:
                import gdown
            except ImportError:
                st.error("gdown non installé.")
                raise
            tmp_zip = DATA_DIR / "data.zip"
            gdown.download(remote_url, str(tmp_zip), quiet=False)
            if not tmp_zip.exists():
                raise RuntimeError("ZIP introuvable après téléchargement.")
            _extract_zip(tmp_zip)
            tmp_zip.unlink(missing_ok=True)
        else:
            import requests

            resp = requests.get(remote_url, timeout=400)
            resp.raise_for_status()
            _extract_zip(resp.content)
    except Exception as e:
        _log(f"Erreur téléchargement: {e}")
        st.error(f"Echec téléchargement: {e}")
        return

    has_raw = any(RAW_DIR.iterdir())
    has_processed = any(PROCESSED_DIR.iterdir())
    _log(f"Après extraction raw={has_raw} processed={has_processed}")
    if not (has_raw and has_processed):
        st.error("Extraction incomplète (raw ou processed vide).")
