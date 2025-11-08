import os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import warnings

# --- Hide FutureWarnings ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def register_students(dataset_path=None):
    """Extract embeddings for all registered students."""
    # Use absolute path (relative to this file)
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

    embeddings = []
    labels = []

    # Check dataset existence
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_path}")

    for student_id in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_id)
        if not os.path.isdir(student_folder):
            continue

        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
                continue

            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                embedding = facenet(face)
                embeddings.append(embedding.detach().cpu().numpy()[0])
                labels.append(student_id)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    return embeddings, labels


if __name__ == '__main__':
    embeddings, labels = register_students()

    # --- Ensure model directory exists ---
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(model_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(model_dir, 'labels.npy'), labels)

    print("✅ Registration complete. Embeddings and labels saved.")
