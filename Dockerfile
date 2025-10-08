# ================================================================
# Dockerfile - Image optimisée QGIS + Flask
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
    gdal-bin \
    libgdal-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5network5 \
    libqt5printsupport5 \
    libqt5svg5 \
    libqt5widgets5 \
    libqt5xml5 \
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
    python3 --version && \
    pip3 --version

# --- Création de la structure de dossiers ---
RUN mkdir -p /opt/render/project/src/data/{shapefiles,csv,geojson,projects,other,tiles,parcels,documents,cache}

# --- Installation des dépendances Python ---
COPY requirements.txt /tmp/requirements.txt

# Installer les dépendances Python
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# --- Copie de l'application ---
WORKDIR /opt/render/project/src
COPY . .

# --- Test QGIS simple ---
RUN python3 -c "from qgis.core import QgsApplication; print('✅ QGIS importable')"

# --- Configuration des permissions ---
RUN chmod -R 755 /opt/render/project/src && \
    chmod -R 777 /opt/render/project/src/data

# --- Exposition du port ---
EXPOSE 10000

# --- Healthcheck ---
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