<div align="center">
  <img width="400" alt="Ilustrasi Model (1)" src="https://github.com/user-attachments/assets/8a6fc1de-dba9-4251-9960-a7b689216467" />
</div>

<h1 align="center">💸 Rupiah Vision: YOLO-Powered Banknote Detector 🤖</h1>

Welcome to **Rupiah Vision**! Ever wondered if a machine could recognize Indonesian money as well as you can? Well, wonder no more! This project uses the power of YOLO (You Only Look Once) for instance segmentation to detect and identify Indonesian Rupiah.

---

## ✨ Key Features

* **High-Speed Detection:** Leverages the efficiency of the YOLO model.
* **Accurate Segmentation:** Precisely outlines each banknote in an image.
* **Wide Range of Classes:** Capable of identifying 7 different Rupiah denominations:
    * 💰 Rp 1.000
    * 💰 Rp 2.000
    * 💰 Rp 5.000
    * 💰 Rp 10.000
    * 💰 Rp 20.000
    * 💰 Rp 50.000
    * 💰 Rp 100.000

---

## 🧠 Dataset & Training Strategy

The robustness of this model comes from a specific training strategy designed to minimize false positives from other currencies.

* **Dataset Composition:** The dataset is composed of 5,000 images, split into training (4,000) and validation (1,000) sets. Each set follows a **50/50 structure**:
    * 💴 **50% Indonesian Rupiah:** A diverse collection of Rupiah banknotes, where each note is carefully labeled with its correct denomination class.
    * 🌍 **50% Other Currencies (Negative Samples):** A wide array of banknotes from various countries (e.g., US Dollar, Euro, Yen).

* **Smart Labeling:** To prevent the model from incorrectly identifying foreign currency as Rupiah, **the images of non-Rupiah currencies were intentionally left unlabeled.** This method teaches the model not only what Rupiah notes look like but, just as importantly, what they *don't* look like. By treating foreign currencies as background noise, the model learns to ignore them, significantly reducing false detections.

---

## 📊 Model Performance

* **Training Data:** 4,000 images 🖼️
* **Validation Data:** 1,000 images 🖼️

| Metric     | Score 📈 |
| :--------- | :------: |
| **mAP50-95** | `0.947`  |
| **Precision**| `0.968`  |
| **Recall** | `0.958`  |

---

## 🛠️ Technology Stack

This project was brought to life using these amazing technologies:

* **Python**
* **PyTorch**
* **YOLO (You Only Look Once)**
* **OpenCV**

---

## 📸 Example Detection

Here’s a sneak peek of the model in action!

<div align="center">
    <img src="https://github.com/user-attachments/assets/0f4f4f19-8bb4-4bca-bbca-584cde708cfe" alt="Example Detection" width="400"/>
</div>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ and lots of ☕
</div>
