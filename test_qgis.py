# test_qgis.py - Test de l'environnement QGIS
import sys
import os
import traceback

print('=== Test environnement QGIS ===')
try:
    from qgis.core import QgsApplication, QgsProject
    print('✅ QGIS core importé avec succès')
    
    # Test d'initialisation basique
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Créer une application QGIS sans interface
    app = QgsApplication([], False)
    app.initQgis()
    print('✅ QgsApplication initialisée')
    
    # Test de projet
    project = QgsProject.instance()
    print('✅ QgsProject fonctionnel')
    
    app.exitQgis()
    print('✅ Environnement QGIS validé')
    
except Exception as e:
    print(f'❌ Erreur QGIS: {e}')
    traceback.print_exc()
    sys.exit(1)