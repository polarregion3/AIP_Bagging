import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 准备数据（以 ACC 为例） =================
acc_data = {
    "Model": ["LightGBM", "SVM", "XGBoost", "MLP", "LogReg", "RandomForest", "ExtraTrees"],
    "handcrafted": [0.5648, 0.3538, 0.5656, 0.4095, 0.5788, 0.5297, 0.4813],
    "prot-T5":    [0.6491, 0.6513, 0.6491, 0.4886, 0.6491, 0.6484, 0.6484],
    "ESM-2":      [0.4484, 0.2967, 0.3458, 0.3941, 0.5795, 0.1832, 0.1736]
}
df_acc = pd.DataFrame(acc_data).set_index("Model")

# ================= 绘制热图 =================
plt.figure(figsize=(6, 5))
sns.heatmap(df_acc, annot=True, fmt=".4f", cmap="YlOrRd", 
            cbar_kws={"label": "Recall Score"})
plt.title("Recall of Different Models on Three Features")
plt.xlabel("Feature Type")
plt.ylabel("Model")
plt.tight_layout()
plt.show()