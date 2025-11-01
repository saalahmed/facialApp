# 🧠 FacialApp

**Reconnaissance faciale via image ou webcam — base locale**

FacialApp est une application **Streamlit** de reconnaissance faciale **100 % locale**.
Elle permet de créer une base de visages à partir d’images, puis de reconnaître des personnes via une image importée ou une capture webcam — **sans envoi de données vers Internet**.

---

## ✨ Fonctionnalités

* 📁 **Base locale** : `faces.json` + dossier `faces/`
* 🧍 **Import multi-images** par personne (détection automatique)
* 📷 **Reconnaissance** par image ou webcam
* 🎚️ **Seuil de correspondance réglable**
* 🖼️ **Image annotée** avec étiquettes `[Nom (distance)]`
* 🗑️ **Suppression** de profils (avec confirmation)
* 🧩 **Formats pris en charge** : `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`, `gif`

---

## 🧱 Prérequis

* **Python 3.9+** (3.12 recommandé)
* Modules requis :

  * `streamlit`
  * `face_recognition`
  * `numpy`
  * `pillow`

> ⚙️ **Pour `face_recognition` :**
>
> * Nécessite `dlib` et ses dépendances (CMake + compilateur C++)
> * Sous **Windows** : [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> * Sous **macOS / Linux** : `gcc` ou `clang`

---

## 🚀 Installation

### 1️ Cloner le dépôt

```bash
git clone https://github.com/saalahmed/DU_Sorbonne_Python-facialApp.git
cd DU_Sorbonne_Python-facialApp
```

### 2 Installer les dépendances

> 💡 Pour éviter les problèmes avec `dlib` et `numpy`, fixe les versions compatibles.

```bash
pip install --upgrade pip setuptools wheel
pip install streamlit==1.51.0 numpy==1.26.4 pillow>=9.5 face-recognition==1.3.0 face_recognition_models==0.3.0
```

---

## ▶️ Lancement

```bash
streamlit run app.py
```

L’application s’ouvre automatiquement sur :
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧭 Utilisation

### 1️⃣ Onglet **Base de données**

* Ajoutez une nouvelle personne → nom + photos.
* Chaque image est :

  * Convertie en RGB JPEG ;
  * Analysée pour extraire une **empreinte faciale** (128 valeurs).

Vous pouvez :

* Rechercher par nom,
* Visualiser les images stockées,
* Supprimer un profil complet.

### 2️⃣ Onglet **Reconnaissance faciale**

* Choisissez une **source** : *Webcam* ou *Upload*.
* Réglez le **seuil** (par défaut : `0.6`) :

  * 👎 Plus petit → plus strict (moins de faux positifs)
  * 👍 Plus grand → plus tolérant
* Cliquez sur 🔎 **Rechercher dans la base** :

  * L’image est annotée avec des cadres verts autour des visages ;
  * Les résultats affichent le **nom**, la **distance**, et le **Top 5** des correspondances.

---

## 🗂️ Structure & Données

```
📦 projet/
 ┣ 📁 faces/              # Dossiers individuels par personne
 ┣ 📄 faces.json          # Base de données principale
 ┣ 📄 app.py              # Application Streamlit
 ┗ 📄 README.md
```

### Exemple d’entrée dans `faces.json`

```json
{
  "id": 1,
  "name": "Marie Dupont",
  "dir": "faces/marie_dupont_1",
  "pictures": [
    {
      "path": "faces/marie_dupont_1/marie.jpg",
      "enc": [/* 128 floats */],
      "faces": 1,
      "box": [top, right, bottom, left]
    }
  ]
}
```

---

## 🔒 Vie privée & conformité

* Les empreintes faciales sont des **données biométriques sensibles**.
* Toutes les données restent stockées **localement** (`faces.json` + images).
* Obtenez toujours le **consentement explicite** des personnes concernées.

---

## 🏗️ Architecture technique

| Composant                   | Rôle                                   |
| --------------------------- | -------------------------------------- |
| **Streamlit**               | Interface utilisateur et navigation    |
| **face_recognition (dlib)** | Encodage et comparaison faciale        |
| **Pillow**                  | Lecture et annotation d’images         |
| **JSON local**              | Stockage des données (`faces.json`)    |
| **Atomic write**            | Sauvegarde sécurisée des modifications |

---

## 🙌 Remerciements

* [Streamlit](https://streamlit.io/)
* [face_recognition](https://github.com/ageitgey/face_recognition)
* La communauté **open source** ❤️

> FacialApp – Reconnaissance faciale 100 % locale et respectueuse de la vie privée.
