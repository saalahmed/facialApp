import streamlit as st
import io, os, json, re, shutil
from pathlib import Path
from typing import Dict
import numpy as np
import face_recognition
from PIL import Image, ImageOps, ImageDraw

# --- Configuration & Consts ---
DB_FILE = "faces.json"
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

SUPPORTED_EXT = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"]
st.set_page_config(page_title="Base de visages", layout="wide")

# --- Utilitaires ---
def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s-]+", "_", text)
    return text or "personne"

def is_supported(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_EXT

def load_json_safe(path: str, default):
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def atomic_write_json(path: str, data: Dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_db() -> Dict:
    data = load_json_safe(DB_FILE, {"people": []})
    return data if isinstance(data.get("people"), list) else {"people": []}

def next_person_id(db: Dict) -> int:
    return max((p.get("id", 0) for p in db["people"]), default=0) + 1

def find_person(db: Dict, name: str):
    name = name.strip().lower()
    return next((p for p in db["people"] if p.get("name", "").strip().lower() == name), None)

def unique_filename(dest: Path, base_stem: str, ext: str = ".jpg") -> str:
    dest.mkdir(parents=True, exist_ok=True)
    stem = slugify(Path(base_stem).stem)
    i, candidate = 1, f"{stem}{ext}"
    while (dest / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate

def safe_rmtree(path: Path):
    base = Path(FACES_DIR).resolve()
    target = Path(path).resolve()
    if target.is_dir() and base in target.parents:
        shutil.rmtree(target)

def pil_open_to_rgb_first_frame(file_like) -> Image.Image:
    """Ouvre n'importe quelle image et la convertit en RGB (8 bits)."""
    if isinstance(file_like, (str, os.PathLike)):
        fp = open(file_like, "rb"); must_close = True
    elif isinstance(file_like, (bytes, bytearray)):
        fp = io.BytesIO(file_like); must_close = True
    else:
        try: file_like.seek(0)
        except Exception: pass
        fp, must_close = file_like, False

    try:
        im = Image.open(fp)
        try: im.seek(0)
        except EOFError: pass
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB": im = im.convert("RGB")
        return im
    finally:
        if must_close: fp.close()

def save_as_rgb_jpeg(uploaded_file, dest_dir: Path) -> str:
    base_stem = Path(getattr(uploaded_file, "name", "image")).stem
    out = dest_dir / unique_filename(dest_dir, base_stem, ".jpg")
    im = pil_open_to_rgb_first_frame(uploaded_file)
    im.save(out, format="JPEG", quality=92, optimize=True, progressive=False)
    return out.as_posix()

# --- Encodages ---
def ensure_rgb_uint8(np_img: np.ndarray) -> np.ndarray:
    """Assure un array 3 canaux uint8 contigu."""
    arr = np_img
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Forme inattendue: {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

def compress_enc(enc, mode="round6"):
    arr = np.array(enc, dtype=np.float32)
    return [round(float(x), 6) for x in arr] if mode=="round6" else arr.astype(np.float16).tolist()

def compute_encoding_from_image(path: str, detection_model: str = "hog"):
    """
    Lit l'image via face_recognition.load_image_file() et renvoie (encoding, nb_faces, box)
    """
    if face_recognition is None:
        return [], 0, None
    np_img = face_recognition.load_image_file(path)
    np_img = ensure_rgb_uint8(np_img)

    locs = face_recognition.face_locations(np_img, number_of_times_to_upsample=1, model=detection_model)
    if not locs:
        return [], 0, None

    encs = face_recognition.face_encodings(np_img, known_face_locations=[locs[0]])
    if not encs:
        return [], len(locs), None

    return encs[0].tolist(), len(locs), locs[0]

def ensure_pic_record(p):
    """Normalise ancien/ancien format (str ou dict) vers dict complet."""
    if isinstance(p, str):
        return {"path": p, "mtime": os.path.getmtime(p) if os.path.exists(p) else None, "enc": [], "faces": None, "box": None}
    return {**{"path": None,"mtime":None,"enc":[],"faces":None,"box":None}, **p}

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt((diff * diff).sum()))

# Collecte toutes les empreintes de la base (et calcule celles manquantes si possible)
def collect_known_encodings(db: Dict, detection_model: str = "hog"):
    known_encs, known_meta = [], []  # meta: (person_id, person_name, pic_path)
    changed = False

    for person in db.get("people", []):
        for pic in person.get("pictures", []):
            rec = ensure_pic_record(pic)
            enc = rec.get("enc") or []
            if not enc and face_recognition is not None and rec.get("path") and os.path.exists(rec["path"]):
                try:
                    new_enc, faces, box = compute_encoding_from_image(rec["path"], detection_model=detection_model)
                    if new_enc:
                        rec["enc"] = compress_enc(new_enc)
                        rec["faces"] = faces
                        rec["box"] = box
                        changed = True
                        # mettre à jour dans la structure d'origine
                        pic.update(rec)
                except Exception:
                    pass
            if rec.get("enc"):
                known_encs.append(np.array(rec["enc"], dtype=np.float32))
                known_meta.append((person["id"], person["name"], rec.get("path")))
    if changed:
        atomic_write_json(DB_FILE, db)
    return np.array(known_encs, dtype=np.float32), known_meta

def encode_runtime_image(img_bytes: bytes, detection_model: str = "hog"):
    """Encode toutes les faces présentes dans l'image donnée (bytes)."""
    if face_recognition is None:
        return [], []
    # Charger via face_recognition pour rester cohérent
    np_img = face_recognition.load_image_file(io.BytesIO(img_bytes))
    np_img = ensure_rgb_uint8(np_img)
    locs = face_recognition.face_locations(np_img, number_of_times_to_upsample=1, model=detection_model)
    if not locs:
        return [], []
    encs = face_recognition.face_encodings(np_img, known_face_locations=locs)
    encs = [np.array(e, dtype=np.float32) for e in encs]
    return locs, encs

def draw_boxes_with_labels(image_rgb: Image.Image, boxes, labels):
    draw = ImageDraw.Draw(image_rgb)
    # Font optionnelle (la police par défaut convient sinon)
    for (top, right, bottom, left), label in zip(boxes, labels):
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        text_w, text_h = draw.textlength(label), 16
        draw.rectangle(((left, bottom), (left + max(40, int(text_w) + 10), bottom + text_h + 6)), fill=(0, 255, 0))
        draw.text((left + 5, bottom + 3), label, fill=(0, 0, 0))
    return image_rgb

# --- Page navigation ---
st.sidebar.title("FacialApp")
page = st.sidebar.radio("Navigation", ["Reconnaissance faciale", "Base de données"])

# --- Session ---
if "uploader_key" not in st.session_state: st.session_state["uploader_key"]=0
if "flash" not in st.session_state: st.session_state["flash"]=None

db = load_db()

# --- Onglet Base ---
if page == "Base de données":
    st.title("Base de données des personnes")

    if st.session_state["flash"]:
        st.success(st.session_state["flash"]); st.session_state["flash"]=None

    with st.form("import_form", clear_on_submit=True):
        st.subheader("Nouvelle personne")
        name = st.text_input("Nom de la personne", placeholder="Ex: Marie Dupont")
        files = st.file_uploader("Photos", type=SUPPORTED_EXT, accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}")
        submitted = st.form_submit_button("Importer", width="stretch")

    if submitted:
        if not name.strip():
            st.error("Merci d'indiquer un nom.")
        elif not files:
            st.error("Merci de sélectionner des images.")
        else:
            person = find_person(db, name)
            if not person:
                pid = next_person_id(db)
                slug = slugify(name)
                person_dir = Path(FACES_DIR)/f"{slug}_{pid}"
                person={"id":pid,"name":name.strip(),"dir":str(person_dir),"pictures":[]}
                db["people"].append(person)
            else:
                person_dir = Path(person["dir"])
            person_dir.mkdir(exist_ok=True, parents=True)

            saved, errors = [], []
            for f in files:
                try:
                    rel_path = save_as_rgb_jpeg(f, person_dir)
                    enc, faces, box = ([], 0, None)
                    if face_recognition is not None:
                        try:
                            enc, faces, box = compute_encoding_from_image(rel_path)
                            if enc: enc = compress_enc(enc)
                        except Exception as e:
                            errors.append(f"{f.name} → encodage impossible: {e}")
                    person["pictures"].append({
                        "path": rel_path, "mtime": os.path.getmtime(rel_path),
                        "enc": enc, "faces": faces, "box": box
                    })
                    saved.append(rel_path)
                except Exception as e:
                    errors.append(f"{f.name} → {e}")
            atomic_write_json(DB_FILE, db)
            msg=[f"{len(saved)} image(s) importée(s) pour **{person['name']}**"]
            if errors: msg.append("⚠️ Erreurs:\n- " + "\n- ".join(errors))
            st.session_state["flash"]="\n\n".join(msg)
            st.session_state["uploader_key"]+=1
            st.rerun()

    q = st.text_input("Recherche une personne", placeholder="Nom...")
    people = [p for p in db["people"] if not q or q.lower() in p["name"].lower()]
    if not people:
        st.info("Aucune personne trouvée.")
    else:
        for p in people:
            pname, pid, pdir = p["name"], p["id"], Path(p["dir"])
            pics = [ensure_pic_record(pic) for pic in p["pictures"]]
            with st.expander(f"👤 {pname} — {len(pics)} image(s)", expanded=False):
                cols = st.columns(min(3,len(pics)) or 1)
                for i, rec in enumerate(pics[:6]):
                    path=rec["path"]
                    with cols[i%len(cols)]:
                        try: st.image(path, caption=Path(path).name, width="stretch")
                        except: st.write(Path(path).name)
                with st.form(f"del_{pid}"):
                    confirm = st.checkbox("Je confirme la suppression", key=f"chk_{pid}")
                    if st.form_submit_button("🗑️ Supprimer", width="stretch"):
                        if confirm:
                            safe_rmtree(pdir)
                            db["people"]=[pp for pp in db["people"] if pp["id"]!=pid]
                            atomic_write_json(DB_FILE, db)
                            st.success("Supprimé."); st.rerun()
                        else:
                            st.warning("Cochez la case de confirmation.")

# --- Onglet Reconnaissance ---
elif page == "Reconnaissance faciale":
    st.title("Reconnaissance de visages")

    if face_recognition is None:
        st.warning(
            "Le module **face_recognition** n'est pas disponible. "
            "Installe-le pour activer l'encodage et la recherche. (dlib + face_recognition)"
        )

    colL, colR = st.columns([1,1])
    with colL:
        source = st.radio("Source", ["Webcam", "Upload"], horizontal=True)
    with colR:
        threshold = st.slider("Seuil de correspondance (plus petit = plus strict)", 0.3, 0.8, 0.6, 0.01)

    # Chargement des encodages connus
    known_encs, known_meta = collect_known_encodings(db)
    if known_encs.size == 0:
        st.info("La base ne contient pas encore d'empreintes exploitables. Ajoute des photos dans l’onglet Base.")
        st.stop()

    img_bytes = None
    if source == "Webcam":
        cam_img = st.camera_input("Prendre une photo")
        if cam_img is not None:
            img_bytes = cam_img.getvalue()
    else:
        up_img = st.file_uploader("Choisir une image", type=SUPPORTED_EXT, key="query_upload")
        if up_img is not None:
            img_bytes = up_img.read()

    do_search = st.button("🔎 Rechercher dans la base", width="stretch", type="primary")

    if do_search:
        if not img_bytes:
            st.error("Aucune image fournie.")
            st.stop()

        # Encode toutes les faces présentes dans l'image requête
        boxes, q_encs = encode_runtime_image(img_bytes, detection_model="hog")
        if not q_encs:
            st.warning("Aucun visage détecté dans l'image fournie.")
            st.stop()

        # Calcul des distances et meilleurs matchs
        results = []  # par visage: dict(box, best_name, best_dist, best_meta, top5)
        for enc, box in zip(q_encs, boxes):
            if face_recognition is not None and hasattr(face_recognition, "face_distance"):
                dists = face_recognition.face_distance(known_encs, enc)
            else:
                # fallback L2
                diffs = known_encs - enc
                dists = np.sqrt(np.sum(diffs*diffs, axis=1))

            order = np.argsort(dists)
            best_idx = int(order[0])
            best_dist = float(dists[best_idx])
            best_meta = known_meta[best_idx]
            best_name = best_meta[1] if best_dist <= threshold else "Inconnu"

            topk = []
            for idx in order[:5]:
                pid, pname, ppath = known_meta[int(idx)]
                topk.append({"name": pname, "distance": float(dists[int(idx)]), "path": ppath, "person_id": pid})

            results.append({
                "box": box,
                "best_name": best_name,
                "best_dist": best_dist,
                "best_meta": best_meta,
                "topk": topk
            })

        # Affichage image annotée
        img_disp = pil_open_to_rgb_first_frame(img_bytes)
        labels = [f"{r['best_name']} ({r['best_dist']:.3f})" for r in results]
        img_annot = draw_boxes_with_labels(img_disp.copy(), [r["box"] for r in results], labels)
        st.image(img_annot, caption="Résultat", width="stretch")

        # Détails par visage
        for i, r in enumerate(results, 1):
            st.subheader(f"Visage #{i}")
            c1, c2 = st.columns([1,1])
            with c1:
                st.markdown(f"**Meilleur match** : {r['best_name']}  \n**Distance** : {r['best_dist']:.4f}  \n**Seuil** : {threshold:.2f}")
            with c2:
                st.markdown("**Top 5 correspondances**")
                for row in r["topk"]:
                    icon = "✅" if row["distance"] <= threshold else "⚠️"
                    st.write(f"{icon} {row['name']} — distance {row['distance']:.4f}")

        st.info("Astuce : diminue le seuil pour être plus strict (moins de faux positifs), ou augmente-le pour être plus tolérant.")
