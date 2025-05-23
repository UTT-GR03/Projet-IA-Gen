# RevivIA
Une application streamlit permettant d'augmenter la résolution d'une image avec des modèles d'IA générative.

## Installation

**Configuration recommandée**
* Python >= 3.7
* PyTorch >= 1.7

**Etapes**
1. Cloner le repo
```
git clone https://github.com/UTT-GR03/Projet-IA-Gen.git
```

2. Installer les packages
```
pip install -r requirements.txt
```
3. Sur Windows, si vous avez créer un environnement virtuel Python `venv`, ouvrir le script `.\venv\Lib\site-packages\basicsr\data\degradations.py`.  
A la ligne 8, remplacer cette ligne de code  
`from torchvision.transforms.functional_tensor import rgb_to_grayscale`  
par  
`from torchvision.transforms.functional import rgb_to_grayscale`



## BibTeX

    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }