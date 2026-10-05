# Enhanced Face Recognition for Surveillance Systems

Face recognition for low-quality surveillance imagery, using GAN-based super
resolution to enhance faces before identification.

## Motivation

Surveillance footage is typically low resolution, poorly lit, and captured at a
distance. Face recognition models trained on clean, high-resolution portraits
degrade sharply under these conditions. This project inserts a super-resolution
enhancement stage between face detection and recognition, and measures whether
the enhancement actually improves identification accuracy rather than just
making images look better.

## How It Works

The pipeline runs in four stages:

1. **Face detection** — Haar cascade classifier (`haarcascade_frontalface_default.xml`)
   locates and crops faces from source frames.
2. **Super-resolution enhancement** — a GAN upscales each low-resolution crop.
   Two approaches are implemented and compared: SR-GAN (`SR_Gan.ipynb`) and
   ESRGAN via BasicSR (`BasicSR_inference_ESRGAN.ipynb`).
3. **Embedding** — an Inception-based FaceNet-style network (`inception_blocks.py`)
   maps each face to an embedding vector.
4. **Identification** — embeddings are matched against an enrolled database
   (`database_utils.py`, `name_in_database.csv`) using a distance threshold.

End-to-end orchestration lives in `Pipeline.ipynb`.

## Repository Structure

| Path | Purpose |
|---|---|
| `Pipeline.ipynb` | End-to-end detection, enhancement, recognition pipeline |
| `preprocessing.ipynb` | Dataset preparation and face cropping |
| `SR_Gan.ipynb` | SR-GAN super-resolution implementation |
| `BasicSR_inference_ESRGAN.ipynb` | ESRGAN inference via BasicSR |
| `Super_resolution_comparison.ipynb` | Side-by-side comparison of SR approaches |
| `inception_blocks.py` | Inception network for face embeddings |
| `facerecog_utils.py` | Recognition helpers, distance computation, thresholding |
| `database_utils.py` | Enrolled-identity database management |
| `utils.py` | Shared I/O and image utilities |
| `haarcascade_frontalface_default.xml` | OpenCV Haar cascade face detector |
| `weights/` | Pretrained model weights |
| `Libraries/` | Vendored dependencies |

## Evaluation Artifacts

| File | Contents |
|---|---|
| `tp_list.csv`, `fp_list.csv`, `tn_list.csv`, `fn_list.csv` | Per-case true/false positive and negative breakdowns |
| `bris_before.csv`, `bris_after.csv` | BRISQUE no-reference image quality scores before and after enhancement |
| `new_gt.npy`, `new_ground_truth.tar.gz` | Ground truth labels |
| `testset.tar.gz`, `new_testset.tar.gz` | Evaluation image sets |

BRISQUE is reported alongside recognition accuracy so that perceptual image
quality improvement can be distinguished from actual identification gains.

## Requirements

<Confirm and pin versions. Inferred from the code layout:>

- Python 3.x
- OpenCV (`opencv-python`) — face detection
- TensorFlow / Keras — Inception embedding network
- PyTorch + BasicSR — ESRGAN inference
- NumPy, Pandas
- A BRISQUE implementation (`<package you used>`)

```bash
pip install -r requirements.txt   # <not yet in repo — worth adding>
