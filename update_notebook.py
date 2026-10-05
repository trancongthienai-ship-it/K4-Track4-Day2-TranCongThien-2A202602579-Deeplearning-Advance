import json

with open("code/lab_day2.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
        
    src = "".join(cell["source"])
    
    if "IMAGES_DIR =" in src:
        cell["source"] = [
            "IMAGES_DIR = \"data/images\"\n",
            "LABELS_DIR = \"data/labels\""
        ]
    elif "đọc split bằng dataset.load_split" in src:
        cell["source"] = [
            "import dataset\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "# 1. Đọc split và check\n",
            "train_df, val_df, test_df = dataset.load_split(LABELS_DIR, fold=0)\n",
            "counts = dataset.check_split(train_df, val_df, test_df, IMAGES_DIR)\n",
            "print(\"Train size:\", counts['train']['total'])\n",
            "print(\"Val size:\", counts['val']['total'])\n",
            "print(\"Test size:\", counts['test']['total'])\n",
            "\n",
            "# 2. Vẽ phân bố lớp\n",
            "train_df['Label'].value_counts().sort_index().plot(kind='bar', title='Class Distribution in Train')\n",
            "plt.show()\n",
            "\n",
            "# 3. Xem thử >= 3 ảnh\n",
            "fig, axes = plt.subplots(3, 3, figsize=(10, 10))\n",
            "for i in range(9):\n",
            "    sample = train_df[train_df['Label'] == i].sample(1).iloc[0]\n",
            "    img = Image.open(f\"{IMAGES_DIR}/{sample['Filename']}\")\n",
            "    axes[i//3, i%3].imshow(img)\n",
            "    axes[i//3, i%3].set_title(dataset.CLASS_NAMES[i])\n",
            "    axes[i//3, i%3].axis('off')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    elif "kiểm tra pipeline" in src:
        cell["source"] = [
            "import train\n",
            "import torch\n",
            "\n",
            "# Cố định seed và kiểm tra pipeline\n",
            "train.set_seed(0)\n",
            "print(\"Seed fixed to 0. Pipeline ready!\")\n"
        ]
    elif "vòng lặp qua danh sách backbone" in src:
        cell["source"] = [
            "from train import Config, run\n",
            "import pandas as pd\n",
            "\n",
            "backbones = ['resnet50', 'convnext_tiny', 'efficientnet_b0', 'swin_tiny', 'deit_small']\n",
            "results_step1 = []\n",
            "\n",
            "for i, b in enumerate(backbones):\n",
            "    print(f\"\\n=== Running Backbone: {b} ===\")\n",
            "    cfg = Config(exp_id=f\"B{i+1:02d}\", backbone=b, seed=0, epochs=3) # Rút gọn epochs để test nhanh\n",
            "    res = run(cfg)\n",
            "    res['backbone'] = b\n",
            "    results_step1.append(res)\n",
            "\n",
            "df_backbones = pd.DataFrame(results_step1)\n",
            "display(df_backbones)\n"
        ]
    elif "các trục A-G của GUIDE.md mục 3" in src:
        cell["source"] = [
            "from train import Config, run\n",
            "\n",
            "best_backbone = 'resnet50' # Chọn thủ công hoặc lấy từ kết quả Bước 1\n",
            "\n",
            "# Trục augmentation (T01)\n",
            "print(\"\\n=== Running Augmentation = RandAug ===\")\n",
            "run(Config(exp_id=\"T01\", backbone=best_backbone, aug=\"randaug\", seed=0, epochs=3))\n",
            "\n",
            "# Trục mix (T02)\n",
            "print(\"\\n=== Running Mixup ===\")\n",
            "run(Config(exp_id=\"T02\", backbone=best_backbone, mix=\"mixup\", mix_alpha=1.0, seed=0, epochs=3))\n",
            "\n",
            "# Trục Loss (T03)\n",
            "print(\"\\n=== Running Focal Loss ===\")\n",
            "run(Config(exp_id=\"T03\", backbone=best_backbone, loss=\"focal\", focal_gamma=2.0, seed=0, epochs=3))\n"
        ]
    elif "TTA lật, multi-crop/scale" in src:
        cell["source"] = [
            "import inference\n",
            "import benchmark\n",
            "import model as mymodel\n",
            "\n",
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "model = mymodel.build_model('resnet50', pretrained=False, num_classes=9).to(device)\n",
            "model.load_state_dict(torch.load(\"runs/T01/seed0/best.pth\"))\n",
            "\n",
            "# Đo độ trễ\n",
            "latency = benchmark.latency_report(model, batch_size=1, img_size=224, dtype=\"fp32\", device=device)\n",
            "print(\"Latency Baseline:\", latency)\n",
            "\n",
            "tta_lat = benchmark.tta_latency(model, k_views=5, batch_size=1, img_size=224, dtype=\"fp32\", device=device)\n",
            "print(\"Latency 5-crop TTA:\", tta_lat)\n"
        ]
    elif "chạy chung kết và mốc" in src:
        cell["source"] = [
            "from train import Config, run\n",
            "\n",
            "# Chọn cấu hình T01 làm chung kết\n",
            "for seed in [0, 1, 2]:\n",
            "    print(f\"\\n=== FINAL RUN - Seed {seed} ===\")\n",
            "    # Chạy chung kết\n",
            "    run(Config(exp_id=\"F01\", backbone=\"resnet50\", aug=\"randaug\", seed=seed, epochs=10, save_test_predictions=True))\n",
            "    # Chạy mốc\n",
            "    run(Config(exp_id=\"T00\", backbone=\"resnet50\", aug=\"basic\", seed=seed, epochs=10, save_test_predictions=True))\n"
        ]
    elif "ghi results.xlsx bằng pandas" in src:
        cell["source"] = [
            "import pandas as pd\n",
            "\n",
            "# Tạo file Excel rỗng với các sheet, sẵn sàng để điền\n",
            "with pd.ExcelWriter(\"results.xlsx\", engine=\"openpyxl\") as writer:\n",
            "    pd.DataFrame(columns=[\"Exp ID\", \"Backbone\", \"Params (M)\", \"GMACs\", \"Macro F1\", \"Time/Epoch\"]).to_excel(writer, sheet_name=\"Backbones\", index=False)\n",
            "    pd.DataFrame(columns=[\"Exp ID\", \"Config Diff\", \"Macro F1\", \"Delta\"]).to_excel(writer, sheet_name=\"Training\", index=False)\n",
            "    pd.DataFrame(columns=[\"Method\", \"Macro F1\", \"Latency p50 (ms)\"]).to_excel(writer, sheet_name=\"Inference\", index=False)\n",
            "    pd.DataFrame(columns=[\"Exp ID\", \"Seed\", \"Test Macro F1\"]).to_excel(writer, sheet_name=\"Final\", index=False)\n",
            "    \n",
            "print(\"Đã tạo form results.xlsx mẫu!\")\n"
        ]

with open("code/lab_day2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
