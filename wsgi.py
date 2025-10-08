# wsgi.py - Point d'entrée WSGI pour Gunicorn
import os
import sys
import logging

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import app, initialize_services

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialiser les services au démarrage
logger = logging.getLogger("qgis-wsgi")

try:
    logger.info("Initialisation des services WSGI...")
    initialize_services()
    logger.info("✅ Services WSGI initialisés avec succès")
except Exception as e:
    logger.error(f"❌ Erreur lors de l'initialisation des services: {e}")
    # Ne pas quitter - l'application peut fonctionner en mode dégradé

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=False)