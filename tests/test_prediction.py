"""
Tests unitaires pour le projet :
GitHub - AlexandruEmil/Data-Science-API-FastAPI-Docker
 
Emplacement recommandé dans le dépôt cloné :
    tests/unit/test_prediction.py
 
Objectif pédagogique :
- Tester uniquement la fonction predict(), sans lancer FastAPI.
- Couvrir les cas nominaux, limites, invalides et exceptionnels.
- Éviter une couverture artificielle basée seulement sur des cas répétitifs.
"""

 
import pytest
 
from app.utils import predict
 
 
# -----------------------------------------------------------------------------
# Cas nominaux : entrées valides et représentatives
# -----------------------------------------------------------------------------
 
@pytest.mark.parametrize(
    "features, expected",
    [
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ([5.0], [10.0]),
        ([1.5, 2.5], [3.0, 5.0]),
    ],
)
 
def test_predict_nominal_cases(features, expected):
    """La fonction doit retourner une prédiction conforme à la règle y = 2x."""
    result = predict(features)
    assert result == pytest.approx(expected)

# -----------------------------------------------------------------------------
# Cas limites : valeurs particulières aux frontières du domaine  
# -----------------------------------------------------------------------------
 
@pytest.mark.parametrize(
    "features, expected",
    [
        ([0.0], [0.0]),
        #([], []),
        ([-1.0], [-2.0]),
        ([-1000.0], [-2000.0]),
        ([1_000_000_0], [2_000_000_0]),
        ([0.0, -1.0, 1_000_000_0], [0.0, -2.0, 2_000_000_0]),
        
    ],
)
 
def test_predict_boundary_cases(features, expected):
    """La fonction doit gérer correctement les valeurs limites ou particulières."""
    result = predict(features)
    
    assert result == pytest.approx(expected)


