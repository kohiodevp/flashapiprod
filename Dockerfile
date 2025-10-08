# ================================================================
# Dockerfile - Image optimisée QGIS + Flask (Version pour Render - Basée sur apt)
# ================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    QGIS_DISABLE_MESSAGE_HOOKS=1 \
    QGIS_NO_OVERRIDE_IMPORT=1 \
    PORT=10000 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- Dépendances système de base ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    # Bibliothèques C/C++ géospatiales
    gdal-bin \
    libgdal-dev \
    # Dépendances Qt pour QGIS
    libqt5core5a \
    libqt5gui5 \
    libqt5network5 \
    libqt5printsupport5 \
    libqt5svg5 \
    libqt5widgets5 \
    libqt5xml5 \
    # Autres dépendances
    fonts-dejavu-core \
    libgl1 \
    libglu1-mesa \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS ---
RUN apt-get update && apt-get install -y \
    qgis \
    qgis-server \
    python3-qgis \
    qgis-providers-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Vérification des installations ---
RUN echo "=== Vérification des installations ===" && \
    qgis --version 2>/dev/null || echo "QGIS CLI non disponible (normal en mode serveur)" && \
    python3 --version && \
    pip3 --version

# --- Création de la structure de dossiers ---
RUN mkdir -p /opt/render/project/src/data/{shapefiles,csv,geojson,projects,other,tiles,parcels,documents,cache}

# --- Installation des dépendances Python ---
COPY requirements.txt /tmp/requirements.txt

# Installer les dépendances Python principales
RUN pip3 install --no-cache-dir \
    Flask==2.3.3 \
    Flask-CORS==4.0.0 \
    Flask-Compress==1.14 \
    gunicorn==21.2.0 \
    PyJWT==2.8.0 \
    passlib==1.7.4 \
    pydantic==1.10.12 \
    redis==4.6.0

# Installer les dépendances géospatiales (versions compatibles avec QGIS système)
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    geopandas==0.13.2 \
    shapely==2.0.1 \
    pyproj==3.6.0 \
    fiona==1.9.4

# --- Copie de l'application ---
WORKDIR /opt/render/project/src
COPY api.py .
COPY wsgi.py .
COPY default.qgs data/projects/default.qgs

# --- Configuration des permissions ---
RUN chmod -R 755 /opt/render/project/src && \
    chmod -R 777 /opt/render/project/src/data

# --- Vérification de l'environnement QGIS ---
RUN python3 -c "\
import sys\n\
print('=== Test environnement QGIS ===')\n\
try:\n\
    from qgis.core import QgsApplication, QgsProject\n\
    print('✅ QGIS core importé avec succès')\n\
    \n\
    # Test d'initialisation basique\n\
    import os\n\
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n\
    \n\
    # Créer une application QGIS sans interface\n\
    app = QgsApplication([], False)\n\
    app.initQgis()\n\
    print('✅ QgsApplication initialisée')\n\
    \n\
    # Test de projet\n\
    project = QgsProject.instance()\n\
    print('✅ QgsProject fonctionnel')\n\
    \n\
    app.exitQgis()\n\
    print('✅ Environnement QGIS validé')\n\
    \n\
except Exception as e:\n\
    print(f'❌ Erreur QGIS: {e}')\n\
    import traceback\n\
    traceback.print_exc()\n\
    sys.exit(1)\
"

# --- Exposition du port ---
EXPOSE 10000

# --- Healthcheck amélioré ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:10000/api/health || exit 1

# --- Démarrage avec Gunicorn ---
CMD ["gunicorn", "--bind", "0.0.0.0:10000", \
     "--workers", "2", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "wsgi:app"]